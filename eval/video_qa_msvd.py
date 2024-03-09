import sys
sys.path.append('./')
import os
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import numpy as np
from model.meta import MetaModel
from data.conversation_lib import conv_templates
from data import video_utils


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]


class CaptionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.datas = json.load(open('datasets/Eval/video/MSVD/MSVD-QA/test_qa.json'))
        map_ids =[x.strip().split(' ') for x in open('datasets/Eval/video/MSVD/MSVD-QA/youtube_mapping.txt').readlines()]
        self.id_to_video_ids = {x[1]:x[0] for x in map_ids}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = 'vid'+str(data['video_id'])
        video_name = self.id_to_video_ids[video_id] + '.avi'
        image_path = os.path.join("datasets/Eval/video/MSVD/YouTubeClips", video_name)
        image = load_video(image_path)
        question_id = data['id']
        question = data['question'] + '\nAnswer the question using a single word or phrase.'
        answer = data['answer']
        return image, question, question_id, answer


if __name__ == "__main__":
    pretrained_path = "path/to/pretrained/ckpt/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_msvd.json"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)    
    
    mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23563")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel("onellm", "config/llama2/7B.json", None, "config/llama2/tokenizer.model")
       
    print("Loading pretrained weights ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    def multi_modal_generate(images, inps):
        images = images.cuda().to(target_dtype)

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=['video'])
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs

    result = {}
    print("Starting...")
    dataset = CaptionDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    predictions = []
    correct = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, question_ids, answers = data
            preds = multi_modal_generate(images, questions)
            for question, pred, question_id, answer in zip(questions, preds, question_ids, answers):
                predictions.append({'question_id': question_id.item(), 'answer': pred, 'gt_answer': answer})
                pred = pred.strip().lower()
                answer = answer.strip().lower()
                if (pred in answer) or (answer in pred):
                    correct += 1
    
    acc = float(correct) / len(dataset)
    print('Accuracy:', acc) 

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)