from typing import Iterable, List
try:
 from petrel_client.client import Client
except:
    print("petrel_client is not installed.")
import json
import torch
from io import BytesIO
import random
import multiprocessing as mp
import copy
from torch.utils.data import Dataset
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model.tokenizer import Tokenizer
import numpy as np
import warnings
import bisect

from .data_utils import make_audio_features, pc_norm, transform_pairimg_train, transform_img_train
from . import video_utils
from .imu_utils import get_imu_frames


DATASETS = dict(
    image=dict(
        train=(
            sorted(list(Path('datasets/Pretrain/image/laion400m_new').glob('*.csv')))[:1000]+
            sorted(list(Path('datasets/Pretrain/image/laion_coco').glob('*.csv')))[:1000]
        ),
        test= ('datasets/Pretrain/image/coco_caps_train2017.csv',),
        max_words=96,
    ),
    audio=dict(
        train=(
            sorted(list(Path('datasets/Pretrain/audio/wavcaps').glob('*.csv')))
        ),
        test=None,
        max_words=96,
    ),
    video=dict(
        train=(
            "datasets/Pretrain/video/webvid/results_2M_train_ceph.csv",),
        test=None,
        max_words=96
    ),
    point=dict(
        train=(
            "datasets/Pretrain/point/pointllm/cap3d_pointllm_train.csv",
        ),
        test=(
            "datasets/Pretrain/point/pointllm/cap3d_pointllm_test.csv",
        ),
        max_words=96,
    ),
    rgbd=dict(
        train=(
            'datasets/Pretrain/image/cc3m.csv',),
        test=None,
        replace_list=['/cc3m/', '/cc3m_depth/'],
        max_words=96,
    ),
    rgbn=dict(
        train=(
            'datasets/Pretrain/image/cc3m.csv',
            ),
        test=None,
        replace_list=['/cc3m/', '/cc3m_normal/'],
        max_words=96,
    ),
    fmri=dict(
        train=(
                "datasets/Pretrain/fmri/nsd/train_sub01.csv",
                "datasets/Pretrain/fmri/nsd/val_sub01.csv",
            ),
        test=(
                "datasets/Pretrain/fmri/nsd/test_sub01.csv",
        ),
        max_words=96
    ),
    imu=dict(
        train=(
            "datasets/Pretrain/imu/ego4d/window_idx_train.json",
        ),
        test=(
            "datasets/Pretrain/imu/ego4d/window_idx_val.json",
        ),
        imu_path="datasets/Pretrain/imu/ego4d/v2/processed_imu/",
        max_words=96,
    )
)


class PretrainDataset(Dataset):
    def __init__(self, dataset='image', partition='train', epochs=1, tokenizer_path=None, petrel_conf=None):
        self.dataset = dataset
        input_filenames = DATASETS[dataset][partition]

        self.petrel_conf = petrel_conf
        self.client = None
        self.partition = partition
        manager = mp.Manager()
        self.datas = manager.list()
        self.captions = manager.list()
        print('loading csv...')
        for input_filename in input_filenames:
            print(input_filename)
            if dataset != 'imu':
                chunk = pd.read_csv(input_filename, sep='\t', on_bad_lines='skip', lineterminator='\n')
                self.datas.extend(chunk['url'].tolist())
                self.captions.extend(chunk['caption'].tolist())
            else:
                self.datas = json.load(open(input_filename))
                self.imu_path = DATASETS[dataset]['imu_path']

        self.max_words = DATASETS[dataset]['max_words']
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.epochs = epochs

    def __len__(self):
        return int(len(self.datas) * self.epochs)
    
    def load_trans_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = transform_img_train(image)
        return image

    def load_trans_image_from_ceph(self, image_path):
        if self.client is None:
            self.client = Client(conf_path=self.petrel_conf)
        image = self.client.get(image_path)

        image = memoryview(image)
        image = Image.open(BytesIO(image)).convert('RGB')
        image = transform_img_train(image)
        return image

    def load_audio(self, audio_path):
        fbank = make_audio_features(audio_path, mel_bins=128, aug=True)
        fbank = fbank.transpose(0, 1)[None] #[1, 128, 1024]
        return fbank
    
    def load_video(self, video_path):
        video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
        return video_feats[:, :, 0]

    def load_video_from_ceph(self, video_path):
        if self.client is None:
            self.client = Client(conf_path=self.petrel_conf)
        video = self.client.get(video_path)
        video = memoryview(video)
        video = BytesIO(video)
        video_feats = video_utils.load_and_transform_video_data(video, video_path, clip_duration=1, clips_per_video=5)
        return video_feats[:, :, 0]

    def load_point(self, point_path):
        point_feat = np.load(point_path)
        # [8196, 6]
        point_feat = torch.tensor(point_feat)
        point_feat = pc_norm(point_feat)

        return point_feat
    
    def load_rgbx(self, image_path):
        replace_list = DATASETS[self.dataset]['replace_list']
        x_image_path = image_path.replace(replace_list[0], replace_list[1])
        image = Image.open(image_path).convert('RGB')
        x_image = Image.open(x_image_path).convert('RGB')
        x_image = x_image.resize(image.size[-2:])

        image, x_image = transform_pairimg_train([image, x_image])

        # [2, 3, H, W]
        image = torch.stack([image, x_image], dim=0)
        return image

    def load_rgbx_from_ceph(self, image_path):
        if self.client is None:
            self.client = Client(conf_path=self.petrel_conf)

        replace_list = DATASETS[self.dataset]['replace_list']
        x_image_path = image_path.replace(replace_list[0], replace_list[1])
        
        image = self.client.get(image_path)
        image = memoryview(image)
        image = Image.open(BytesIO(image)).convert('RGB')

        x_image = self.client.get(x_image_path)
        x_image = memoryview(x_image)
        x_image = Image.open(BytesIO(x_image)).convert('RGB')
        
        x_image = x_image.resize(image.size[-2:])

        image, x_image = transform_pairimg_train([image, x_image])

        # [2, 3, H, W]
        image = torch.stack([image, x_image], dim=0)
        return image
    
    def load_fmri(self, fmri_path):
        data = np.load(fmri_path)
        data = data.mean(axis=0)
        data = torch.tensor(data[None])
        return data

    def load_imu(self, data_dict):
        uid = data_dict["video_uid"]
        w_s = data_dict["window_start"]
        w_e = data_dict["window_end"]

        imu_data = get_imu_frames(
            self.imu_path, uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
        )
        if imu_data is None:
            raise ValueError
        return imu_data['signal']

    def __getitem__(self, index):
        index = index % len(self.datas)
        if self.dataset != 'imu':
            data_path, caption = self.datas[index], self.captions[index]
        else:
            data_dict = self.datas[index]
            caption = data_dict['text']
            data_path = data_dict["video_uid"]

        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        try:
            if self.dataset == 'image':
                data = self.load_trans_image_from_ceph(data_path)
            elif self.dataset == 'audio':
                data = self.load_audio(data_path)
            elif self.dataset == 'video':
                data = self.load_video_from_ceph(data_path)
            elif self.dataset == 'point':
                data = self.load_point(data_path)
            elif self.dataset in ['rgbn', 'rgbd']:
                data = self.load_rgbx(data_path)
            elif self.dataset == 'fmri':
                data = self.load_fmri(data_path)
            elif self.dataset == 'imu':
                data_dict = self.datas[index]
                data = self.load_imu(data_dict)
        except:
            print(data_path, 'Not Found')
            rand_idx = random.randint(0, len(self))
            return self.__getitem__(rand_idx)  
        
        caption_tokens = torch.tensor(self.tokenizer.encode(caption, bos=True, eos=True), dtype=torch.int64)
        input_data = caption_tokens

        padding = self.max_words - input_data.shape[0]
        if padding > 0:
            input_data = torch.cat((input_data, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            input_data = input_data[:self.max_words]
        labels = copy.deepcopy(input_data)

        if self.partition != 'train':
            return input_data, labels, data, data_path, self.dataset, caption
        
        return input_data, labels, data, data_path, self.dataset

    def __repr__(self):
        return f"<XTextPair:{self.dataset}"


class ConcatDataset(Dataset):
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        # for d in self.datasets:
            # assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def get_indices(self, batch_size, world_size=1, rank_id=0):
        random.seed(0)
        real_batch_size = batch_size * world_size
        batch_train_indices = []
        num_batches = []
        for i in range(len(self.datasets)):
            # get train_indices
            start_idx = self.cumulative_sizes[i-1] if i>0 else 0
            end_idx = self.cumulative_sizes[i]
            train_indice = list(range(start_idx, end_idx))
            random.shuffle(train_indice)
            num_batch = int(len(self.datasets[i]) / real_batch_size)
            num_batches.append(num_batch)
            # get batch indices for each rank
            batch_train_indice = [
                train_indice[batch*real_batch_size:(batch+1)*real_batch_size][rank_id::world_size]
                for batch in range(num_batch)
            ]
            batch_train_indices.append(batch_train_indice)
        min_num_batch = min(num_batches)

        train_indices = []
        for batch in range(min_num_batch):
            for i in range(len(self.datasets)):
                train_indices.extend(batch_train_indices[i][batch])
        
        return train_indices