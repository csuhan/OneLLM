import argparse
import datetime
import numpy as np
import os
import copy
import time
from pathlib import Path
import functools
import multiprocessing

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from fairscale.nn.model_parallel import initialize as fs_init

try:
    from apex.optimizers import FusedAdam as AdamW
except ImportError:
    warnings.warn("cannot import FusedAdam from apex, use torch AdamW instead")
    from torch.optim import AdamW

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from model.meta import MetaModel
from engine_pretrain import train_one_epoch

import warnings
warnings.filterwarnings("ignore")

from data.pretrain_dataset import PretrainDataset, ConcatDataset


def get_args_parser():
    parser = argparse.ArgumentParser('OneLLM Pretraining', add_help=False)
    parser.add_argument('--datasets', type=str, default='image', nargs='+')
    parser.add_argument('--epochs', default=1, type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--llama_ckpt_dir", type=str, default="")
    parser.add_argument("--llama_config", type=str, default="config/llama2/7B.json")
    parser.add_argument("--tokenizer_path", type=str, default="config/llama2/tokenizer.model")
    parser.add_argument("--petrel_conf", type=str, default="")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_iters', type=int, default=20000, metavar='N',
                        help='iterations to warmup LR')
    parser.add_argument('--lr_decay_iters', type=int, default=1800000, metavar='N',
                        help='iters before keeping minimal learning rate')

    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--init_from', default='',
                        help='init from checkpoint')
    parser.add_argument('--init_from_image', action='store_true')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['ddp', 'sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--save_consolidated', action="store_true",
                        help="save consolidated model weights along with regular checkpoints "
                             "used to resume training. useful for convenient deployment but "
                             "will occupy some additional disk space.")
    parser.add_argument("--checkpointing", action="store_true")

    return parser


def main(args):
    multiprocessing.set_start_method("spawn")
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True

    global_rank = misc.get_rank()
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_group = fs_init.get_data_parallel_group()


    datasets_train = {
        dataset: PretrainDataset(dataset=dataset, partition='train', epochs=epoch, tokenizer_path=args.tokenizer_path, petrel_conf=args.petrel_conf)
        for dataset, epoch in zip(args.datasets, args.epochs)
    }
    print("Length of each dataset:", {dataset_name: len(dataset) for dataset_name, dataset in datasets_train.items()})

    datasets_train = ConcatDataset(list(datasets_train.values()))
    train_indices = datasets_train.get_indices(args.batch_size, world_size=dp_world_size, rank_id=dp_rank)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    model = MetaModel(args.llama_type, args.llama_config, args.llama_ckpt_dir, args.tokenizer_path)
    model.to(device)
    print("Model = %s" % str(model))
    if args.init_from:
        print("Init checkpoint from %s" % args.init_from)
        ckpt_path = os.path.join(args.init_from, f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.init_from, f"consolidated.{mp_rank:02d}.pth")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            checkpoint = {'llma.'+key:val for key,val in checkpoint.items()}
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
        if args.init_from_image:
            for expert_id in range(model.llma.num_experts):
                expert_id = str(expert_id)
                if not any([f'llma.resample_layers.{expert_id}' in key for key in checkpoint]):
                    print(f'init {expert_id} projection weight from image')
                    model.llma.resample_layers[expert_id].load_state_dict({k.replace('llma.resample_layers.image.', ''):v for k,v in checkpoint.items() if 'llma.resample_layers.image' in k})
                else:
                    print(f'init {expert_id} projection weight from Pretrain model')

            for modal in model.llma.modals:
                if not any([modal in key for key in checkpoint]):
                    print(f'init {modal} clip_proj & tag weight from image')
                    dtype = model.llma.resample_tokens[modal].data.dtype
                    model.llma.resample_tokens[modal].data = checkpoint['llma.resample_tokens.image'].to(device, dtype=dtype)
                    model.llma.clip_proj1[modal].load_state_dict({k.replace('llma.clip_proj1.image.', ''):v for k,v in checkpoint.items() if 'llma.clip_proj1.image' in k})
                    model.llma.clip_proj2[modal].load_state_dict({k.replace('llma.clip_proj2.image.', ''):v for k,v in checkpoint.items() if 'llma.clip_proj2.image' in k})
                    model.llma.start_tag[modal].data = checkpoint['llma.start_tag.image'].to(device, dtype=dtype)
                    model.llma.end_tag[modal].data = checkpoint['llma.end_tag.image'].to(device, dtype=dtype)
                else:
                    print(f'init {modal} clip_proj & tag weight from Pretrain model')

    mixed_precision_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }[args.precision]
    TransformerBlock = type(model.llma.layers[0])
    model = FSDP(
        model,
        process_group=fs_init.get_data_parallel_group(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[TransformerBlock],
        ),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        ),
        sharding_strategy={
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "ddp": ShardingStrategy.NO_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
        }[args.data_parallel],
        ignored_parameters=[param for param in model.parameters() if not param.requires_grad],
    )

    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    eff_batch_size = args.batch_size * args.accum_iter * fs_init.get_data_parallel_world_size()
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    #param_groups = misc.add_weight_decay(model, args.weight_decay)
    param_groups = {
        "decay": {"params": [], "weight_decay": args.weight_decay, "lr": args.lr},
        "no_decay": {"params": [], "weight_decay": 0., "lr": args.lr},
        "scratch_decay": {"params": [], "weight_decay": args.weight_decay, "lr": args.lr},
        "scratch_no_decay": {"params": [], "weight_decay": 0., "lr": args.lr},
    }
    print("Making parameter groups ...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = name.endswith(".bias") or name.endswith("norm.weight")
        scratch = "llma.resample_layers" in name or "llma.resample_tokens" in name
        group_name = ("scratch_" if scratch else "") + ("no_decay" if no_decay else "decay")
        print(f"{name}: in group {group_name}")
        param_groups[group_name]["params"].append(param)
    optimizer = AdamW(
        [param_groups[key] for key in ["decay", "no_decay", "scratch_decay", "scratch_no_decay"]],
        betas=(0.9, 0.95),
    )
    print(optimizer)
    loss_scaler = NativeScaler(args)

    start_iter = 0
    if args.resume or args.auto_resume:
        _, start_iter = misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
        train_indices = train_indices[start_iter * args.batch_size:]
    
    data_loaders_train = torch.utils.data.DataLoader(
        datasets_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=16,
        sampler=train_indices)

    print(f"Start training")
    start_time = time.time()

    train_stats = train_one_epoch(
        model, data_loaders_train,
        optimizer, device, 0, start_iter, loss_scaler,
        log_writer=log_writer,
        args=args
    )

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
