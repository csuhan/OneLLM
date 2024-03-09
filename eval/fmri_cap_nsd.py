import sys
sys.path.append('./')
import json
import os
import torch
from model.meta import MetaModel
import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import numpy as np
import torch.distributed as dist
from data.conversation_lib import conv_templates


def load_fmri(fmri_path):
    data = np.load(fmri_path)
    data = data.mean(axis=0)
    data = torch.tensor(data[None])
    return data

class CaptionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.fmri_anns = json.load(open("datasets/Eval/fmri/fmri_eval_cococap.json"))
        self.fmri_ids = [x['id'] for x in self.fmri_anns['images']]
        self.fmri_names = [x['file_name'] for x in self.fmri_anns['images']]
        self.fmri_files = self.fmri_names
    
    def __len__(self):
        return len(self.fmri_files)
      
    def __getitem__(self, index):
        fmri_file = self.fmri_files[index]
        return load_fmri(fmri_file), self.fmri_names[index], self.fmri_ids[index]


if __name__ == "__main__":
    pretrained_path = "path/to/pretrained/ckpt/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_fmricap.json"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23560")
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

    def multi_modal_generate(images, inps, modal=['image']):
        images = images.cuda().to(target_dtype)

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=modal)
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs
    
    dataset = CaptionDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    outputs = []
    for data in tqdm.tqdm(dataloader):
        fmris, fmri_names, fmri_ids = data
        prompts = ['Provide a one-sentence caption for the provided fMRI data.'] * len(fmris)

        results = multi_modal_generate(fmris, prompts, modal=['fmri'])

        for fmri_name, fmri_id, result in zip(fmri_names, fmri_ids, results):
            outputs.append({
                'image_id': fmri_id.item(),
                'caption': result.strip()
            })
            print(fmri_name, fmri_id, result.strip())
            print('='*10)
    
    with open(answer_path, 'w') as f:
        json.dump(outputs, f)
