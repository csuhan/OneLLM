#!/bin/bash
#SBATCH -p llmeval2
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:8
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --cpus-per-task=14
#SBATCH --requeue
#SBATCH --open-mode append
#SBATCH -x SH-IDC1-10-140-1-[1-100]

# SBATCH -w SH-IDC1-10-140-1-[1118,134-135,137]

# export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0//bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0//lib/:/mnt/petrelfs/share/gcc/gcc-7.5.0//lib64:/mnt/petrelfs/share/nccl_2.12.12-1+cuda11.0_x86_64/lib/:$LD_LIBRARY_PATH

export PYTHONPATH=/mnt/petrelfs/share_data/linziyi/petrel-oss-python-sdk/build/lib:$PYTHONPATH

warmup=2000
max_iter=500000
accum=8
llama_type=onellm
lr=2e-5
minlr=5e-6
grad_clip=2

exp_name="onellm_llama2-7B_img224-patch16_${llama_type}_bsz512-5120_lr${lr}_warm${warmup}_clip${grad_clip}_X_v20"
mkdir -p multimodal_llama2_7B/"$exp_name"

# srun -p alpha_vl --gres=gpu:1 --cpus-per-task 16 -n1 \
# --ntasks-per-node=1 --quotatype=spot \
srun python -u main_pretrain.py \
--epochs 1 50 30 7 7 2017 23 8 \
--dataset image audio point rgbd rgbn fmri imu video \
--batch_size 10 --accum_iter ${accum} \
--model_parallel_size 1 \
--data_parallel sdp \
--save_consolidated \
--llama_type ${llama_type} \
--petrel_conf /mnt/lustre/hanjiaming/petrelfs/code/MM-LLM/petreloss_all.conf \
--init_from "multimodal_llama2_7B/WeMix-7B-clip_resampler_iter210000_new" \
--init_from_image \
--auto_resume \
--weight_decay 0.1 --output_dir onellm_ckpt/"$exp_name" \
--warmup_iters ${warmup} --lr_decay_iters ${max_iter} --lr ${lr} --min_lr ${minlr} --clip_grad ${grad_clip} \
--save_freq 1000 \
2>&1 | tee -a multimodal_llama2_7B/"$exp_name"/output.log
