import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import torchvision.transforms as transforms
from model.meta import MetaModel
from data.conversation_lib import conv_templates

    
T_resized_center_crop = transforms.Compose([
    transforms.Resize(
        336, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


if __name__ == "__main__":
    pretrained_path = "path/to/pretrained/ckpt/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_mmvet.json"
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

    def multi_modal_generate(img_path, inp):
        
        conv = conv_templates["v1"].copy()
        if img_path is not None:
            image = Image.open(img_path).convert('RGB')
            image = T_resized_center_crop(image).unsqueeze(0).cuda().to(target_dtype)
        else:
            image = None
    
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        with torch.cuda.amp.autocast(dtype=target_dtype):
            response = model.generate([conv.get_prompt()], image, 256, temperature=0.1, top_p=0.75, modal=['image'])
            response = response[0]
            response = response[len(conv.get_prompt()):].split('###')[0]
            conv.messages[-1][-1] = response
            return response.strip()

    result = {}
    batch_size = 1
    print("Starting...")    
    datas = json.load(open('datasets/Eval/image/mm-vet/mm-vet.json'))
    predictions = {}
    with torch.no_grad():
        for image_name, data in tqdm(datas.items()):
            image_path = os.path.join('datasets/Eval/image/mm-vet/images', data['imagename'])
            pred = multi_modal_generate(image_path, data['question'])
            predictions[image_name]=pred

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)