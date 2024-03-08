import warnings

from typing import Optional
import os
import copy
import numpy as np
import random
from torch.utils.data import Sampler, Dataset
import torch
from PIL import Image
import json
from model.tokenizer import Tokenizer
from . import conversation_lib

from . import video_utils
from .imu_utils import get_imu_frames
from .data_utils import T_random_resized_crop, transform_pairimg_train, make_audio_features

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


class ConversationGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = f"{conversation_lib.default_conversation.system}\n\n"
        self._probe_tokenizer_style()

    def _probe_tokenizer_style(self):
        """
        Given a sentence, e.g. "My darling", some tokenizers will make the space a seperate token,
        while some others will merge the space into the next word, forming a token representing " darling".
        Knowing which style the tokenizer takes is necessary for correct ground-truth label masking.

        """
        probe = "Probe am I"
        sentence1 = self.tokenizer.encode(conversation_lib.default_conversation.roles[1] + ": " + probe,
                                          bos=False, eos=False)
        sentence2 = self.tokenizer.encode(probe,
                                          bos=False, eos=False)
        if sentence1[-len(sentence2):] == sentence2:
            self.space_before_to_predict = False
        else:
            sentence3 = self.tokenizer.encode(" " + probe,
                                              bos=False, eos=False)
            assert sentence1[-len(sentence3):] == sentence3
            self.space_before_to_predict = True

    def add_speaker_and_signal(self, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = self.header

        to_predict_list = []

        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() in ["human"]:
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() in ["gpt", "assistant"]:
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]
            if DEFAULT_IMAGE_TOKEN in value:
                value = value.replace(DEFAULT_IMAGE_TOKEN, '').strip()

            sentence_value = BEGIN_SIGNAL + from_str + ": " + value + END_SIGNAL

            if from_str == conversation_lib.default_conversation.roles[1]:
                to_predict_value = value + END_SIGNAL + "###"
                if self.space_before_to_predict:
                    to_predict_value = " " + to_predict_value
                to_predict_list.append(to_predict_value)

            if get_conversation:
                conversation = conversation + sentence_value

        conversation = conversation + BEGIN_SIGNAL
        return conversation, to_predict_list


DATASETS = dict(
    image=[
        dict(path="datasets/InstructionTuning/image/llava_v1_5_mix665k_image.json", type='image'),
        dict(path='datasets/InstructionTuning/image/cococap_train.json', type='image'),
        dict(path="datasets/InstructionTuning/image/llava_v1_5_mix665k_text.json", type='text'),
    ],
    audio=[
        dict(path="datasets/InstructionTuning/audio/audiocap_train.json", type='audio'),
        dict(path="datasets/InstructionTuning/audio/audiocap_val.json", type='audio'),
        dict(path="datasets/InstructionTuning/audio/audio_conversation.json", type='audio'),
    ],
    video=[
        dict(path="datasets/InstructionTuning/video/msrvtt_cap_trainval.json", type='video'),
        dict(path="datasets/InstructionTuning/video/msrvtt_cap_test.json", type='video'),
        dict(path="datasets/InstructionTuning/video/msrvtt_vqa_train.json", type='video'),
        dict(path="datasets/InstructionTuning/video/msrvtt_vqa_val.json", type='video'),
        dict(path="datasets/InstructionTuning/video/msrvtt_vqa_test.json", type='video'),
        dict(path="datasets/InstructionTuning/video/video_complex_reasoning_10k.json", type='video'),
        dict(path="datasets/InstructionTuning/video/video_conversation_10k.json", type='video'),
        dict(path="datasets/InstructionTuning/video/video_detail_10k.json", type='video'),
    ],
    point=[
        dict(path="datasets/InstructionTuning/point/pointllm_70k.json", type='point'),
    ],
    rgbd=[
        dict(path="datasets/InstructionTuning/depth_normal/llava_instruct_50k_depth.json", type='rgbd'),
    ],
    rgbn=[
        dict(path="datasets/InstructionTuning/depth_normal/llava_instruct_50k_normal.json", type='rgbn'),
    ],
    imu=[
        dict(path="datasets/InstructionTuning/imu/imu_fixed_50k.json", type='imu'),
    ],
    fmri=[
        dict(path="datasets/InstructionTuning/fmri/fmri_fixed_train.json", type='fmri'),
    ],
)

class FinetuneDialogDataset(Dataset):
    def __init__(self, dataset=['image'], transform=T_random_resized_crop, max_words=2048, image_words=30, tokenizer_path=None):
        if isinstance(dataset, str):
            dataset = [dataset]
        self.dataset = dataset

        group_ann = {}
        for d in dataset:
            for meta in DATASETS[d]:
                meta_path, meta_type = meta['path'], meta['type']
                meta_ext = os.path.splitext(meta_path)[-1]
                if meta_ext == ".json":
                    with open(meta_path) as f:
                        meta_l = json.load(f)
                        # add data_type
                        # this is a temp solution
                        new_meta_l = []
                        for l in meta_l:
                            l['data_type'] = meta_type
                            new_meta_l.append(l)
                        meta_l = new_meta_l
                elif meta_ext == ".jsonl":
                    meta_l = []
                    with open(meta_path) as f:
                        for i, line in enumerate(f):
                            try:
                                meta_l.append(json.loads(line))
                            except json.decoder.JSONDecodeError as e:
                                print(
                                    f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                                raise e
                else:
                    raise NotImplementedError(
                        f"Unknown meta file extension: \"{meta_ext}\". "
                        f"Currently, .json, .jsonl are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
                if meta_type not in group_ann:
                    group_ann[meta_type] = []
                print(f"{meta_path}, type {meta_type}: len {len(meta_l)}")
                group_ann[meta_type] += meta_l

        # sort group_ann for higher efficiency (items in one global batch with similar length)
        for meta_type, meta_l in group_ann.items():
            meta_l.sort(key=lambda data_item: sum(
                [len(_['value']) for _ in data_item['conversations']]))

        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(
                range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        print(f"total length: {len(self)}")
        self.transform = transform
        print(f"transform:\n{self.transform}")
        self.max_words = max_words
        self.image_words = image_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.conversation_generator = ConversationGenerator(self.tokenizer)

        self.load_funcs = dict(
            image=self.load_image,
            audio=self.load_audio,
            video=self.load_video,
            point=self.load_point,
            rgbd=self.load_rgbx,
            rgbn=self.load_rgbx,
            imu=self.load_imu,
            fmri=self.load_fmri
        )

    def __len__(self):
        return len(self.ann)

    def load_image(self, data):
        filename = data['image']
        image = Image.open(filename).convert('RGB')
        image = self.transform(image)
        return image

    def load_audio(self, data):
        audio_path = data['image']
        fbank = make_audio_features(audio_path, mel_bins=128)
        fbank = fbank.transpose(0, 1)[None]  # [1, 128, 1024]
        return fbank

    def load_video(self, data):
        video_path = data['image']
        video_feats = video_utils.load_and_transform_video_data(
            video_path, video_path, clip_duration=1, clips_per_video=5)
        return video_feats[:, :, 0]

    def load_point(self, data):
        point_path = data['image']
        point_feat = torch.load(point_path, map_location='cpu')
        point_feat = point_feat.transpose(0, 1)
        return point_feat

    def load_rgbx(self, data):
        image_path = data['image']
        x_image_path = data['depth_image'] if 'depth_image' in data else data['normal_image']
        image = Image.open(image_path).convert('RGB')
        x_image = Image.open(x_image_path).convert('RGB')
        x_image = x_image.resize(image.size[-2:])

        image, x_image = transform_pairimg_train([image, x_image])
        # [2, 3, H, W]
        image = torch.stack([image, x_image], dim=0)
        return image

    def load_fmri(self, data):
        fmri_path = data['image']
        data = np.load(fmri_path)
        data = data.mean(axis=0)
        data = torch.tensor(data[None])
        return data

    def load_imu(self, data_dict):
        imu_path = data_dict['imu_path']
        uid = data_dict["video_uid"]
        w_s = data_dict["window_start"]
        w_e = data_dict["window_end"]

        imu_data = get_imu_frames(
            imu_path, uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
        )
        if imu_data is None:
            raise ValueError
        return imu_data['signal']

    def __getitem__(self, index, expect_type=None):
        if expect_type is None:
            data_item = self.ann[index]
        else:
            # in case we want get data from specific data_type
            data_item = self.group_ann[expect_type][index]

        data_type = data_item['data_type']
        if data_type != 'text':
            if data_type in self.load_funcs:
                try:
                    image = self.load_funcs[data_type](data_item)
                    if image == None:
                        raise ValueError('Data is None')
                except:
                    print('Error', data_item)
                    rand_idx = random.randint(
                        0, len(self.group_ann[data_type]))
                    return self.__getitem__(rand_idx, expect_type=data_type)
            else:
                raise ValueError(f'Does not support {data_type}')
        else:
            image = None
            # warnings.warn("pure black image for examples without image")
            # image = torch.zeros(3, 224, 224)

        source = data_item["conversations"]
        conversation, to_predict_values = self.conversation_generator.add_speaker_and_signal(
            source)
        if len(to_predict_values) == 0:
            warnings.warn(
                f"see dialog data with nothing to predict, data: {data_item}")
            return self[index-1]

        tokenzed_conversation = self.tokenizer.encode(
            conversation, bos=True, eos=True)
        labels = [IGNORE_INDEX for _ in tokenzed_conversation]

        check_pos = 0
        for value in to_predict_values:
            tokenized_value = self.tokenizer.encode(
                value, bos=False, eos=False)
            value_pos = find_sublist(
                tokenzed_conversation[check_pos:], tokenized_value) + check_pos
            if value_pos == -1:
                print(
                    "a sentence mismatches the corresponding piece in the conversation")
                return self[index-1]
            labels[value_pos:value_pos+len(tokenized_value)] = tokenized_value
            assert labels[value_pos:value_pos+len(
                tokenized_value)] == tokenzed_conversation[value_pos:value_pos+len(tokenized_value)]
            check_pos = value_pos+len(tokenized_value)

        input2 = torch.tensor(tokenzed_conversation, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words
        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat(
                (input2, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat(
                (labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            labels = labels[:max_words]

        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        if image is None:
            return input2, labels, data_item['data_type']
        else:
            return input2, labels, image, data_item['data_type']

    def groups(self):
        return list(self.group_indices.values())


def find_sublist(a: list, b: list):
    len_a, len_b = len(a), len(b)
    for i in range(len_a - len_b + 1):
        if a[i:i+len_b] == b:
            return i
    return -1


class FinetuneDistSampler(Sampler):
    #   Distrubuted Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0
        self.start_iter = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_)//batch_size for _ in group_indices]
        assert all([_%num_replicas==0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)

        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        global_batch_size = self.batch_size * self.num_replicas * self.acc_grad
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            # for _ in group_indices_shuffle:
            #     rng.shuffle(_)
            global_batched_indices = [
                indices_in_group[i:i+global_batch_size]
                for indices_in_group in group_indices_shuffle
                for i in range(0, len(indices_in_group), global_batch_size)]
            rng.shuffle(global_batched_indices)
            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = [_ for batch_indices in group_indices for _ in batch_indices]

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size:]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter

