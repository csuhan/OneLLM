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
from data.data_utils import pc_norm


class CaptionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.anns = json.load(open('datasets/Eval/point/pointllm_test.json'))
        self.ids = list(self.anns.keys())

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        id = self.ids[index]
        caption = self.anns[id]

        file_path = f'datasets/Eval/point/pointllm/8192_npy/{id}_8192.npy'
    
        point_feat = np.load(file_path)
        point_feat = torch.tensor(point_feat)
        point_feat = pc_norm(point_feat)

        question = 'What is this?'
        answer = caption
        return point_feat, question, id, answer


if __name__ == "__main__":
    pretrained_path = "path/to/pretrained/ckpt/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_pointllm_cap.json"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    
    mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23581")
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
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=['point'])
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs

    print("Starting...")
    dataset = CaptionDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, ids, answers = data
            preds = multi_modal_generate(images, questions)

            for question, pred, id, answer in zip(questions, preds, ids, answers):
                predictions.append({
                    'object_id': id,
                    'model_output': pred,
                    'ground_truth': answer
                })

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)
