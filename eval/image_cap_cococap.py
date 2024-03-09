import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import torchvision.transforms as transforms
from model.meta import MetaModel
from data.conversation_lib import conv_templates


T_resized_center_crop = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


class CocoCapDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.datas = json.load(open('datasets/Eval/image/coco_cap/coco_karpathy_val.json'))
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        image_path = os.path.join("datasets/InstructionTuning/image/coco/", data['image'])
        image = Image.open(image_path).convert('RGB')
        image = T_resized_center_crop(image)
        image_id = int(data['image'].split('_')[-1].split('.')[0])
        question = 'Provide a one-sentence caption for the provided image.'
        return image, question, image_id


if __name__ == "__main__":
    pretrained_path = "path/to/pretrained/ckpt/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_cococap.json"
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

    def multi_modal_generate(images, inps):
        images = images.cuda().to(target_dtype)

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=['image'])
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs

    print("Starting...")
    dataset = CocoCapDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, image_ids = data
            preds = multi_modal_generate(images, questions)
            for question, pred, image_id in zip(questions, preds, image_ids):
                predictions.append({'image_id': image_id.item(), 'caption': pred})

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)