import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
import argparse
import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import setup_for_distributed
from util.misc import default_tensor_type
from model.meta import MetaModel
from data.conversation.lib import conv_templates
from PIL import Image
import torchvision.transforms as transforms


T_random_resized_crop = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=3,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def model_worker(args: argparse.Namespace) -> None:
    rank = 0
    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)

    # set the print behavior.
    setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel(args.llama_type, args.llama_config, tokenizer_path=args.tokenizer_path)
    print("Loading pretrained weights ...")
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.cuda()
    model.eval()
    print(f"Model = {str(model)}")

    print('Model is ready. Please input')

    conv = conv_templates["v1"].copy()

    image = Image.open(args.image_path).convert('RGB')
    image = T_random_resized_crop(image).unsqueeze(0).cuda().to(target_dtype)
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{conv.roles[1]}: ", end="")

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        with torch.cuda.amp.autocast(dtype=target_dtype):
            print(conv.get_prompt())
            response = model.generate([conv.get_prompt()], image, 256, temperature=0.1, top_p=0.75, modal=["image"])
            response = response[0]
            response = response[len(conv.get_prompt()):].split('###')[0]
            print(response)
            conv.messages[-1][-1] = response


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLaMA2-Accessory Chat Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image_path", type=str, help="path to the input image")
    group.add_argument(
        "--gpu_ids", type=int, nargs="+",
        help="A list of space-separated gpu ids to run the model on. "
             "The model will span across GPUs in tensor-parallel mode."
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="config/llama2/tokenizer.model",
        help="Path to the tokenizer.model file provided along with the LLaMA "
             "model."
    )
    parser.add_argument(
        "--llama_type", default="onellm", type=str, metavar="MODEL",
        help="LLaMA model type."
    )
    parser.add_argument(
        "--llama_config", type=str, required=True,
        default="config/llama2/7B.json",
        help="Path to the llama model config json."
    )
    parser.add_argument(
        "--model_max_seq_len", type=int, default=2048,
        help="Max sequence length accepted by the pretrained model."
    )
    parser.add_argument(
        "--pretrained_path", type=str, required=True,
        help="Path to the llama model checkpoints. A list of checkpoints is "
             "supported and will be merged from left to right.")
    parser.add_argument(
        "--master_port", type=int, default=23862,
        help="A port used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--master_addr", type=str, default="127.0.0.1",
        help="An address used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16"], default="fp16",
        help="The dtype used for model weights and inference."
    )
    args = parser.parse_args()

    # check and setup gpu_ids to use
    if args.gpu_ids is None:
        if args.n_gpus is None:
            args.n_gpus = 1
        assert args.n_gpus > 0, (
            "The demo currently must run on a positive number of GPUs."
        )
        args.gpu_ids = list(range(args.n_gpus))

    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    mp.set_start_method("spawn")
    model_worker(args)
