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

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0//bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0//lib/:/mnt/petrelfs/share/gcc/gcc-7.5.0//lib64:$LD_LIBRARY_PATH

export PYTHONPATH=/mnt/petrelfs/share_data/linziyi/petrel-oss-python-sdk/build/lib:$PYTHONPATH


epochs=1
warmup=0.05
accum=4
llama_type=onellm
lr=2e-5
minlr=0.0
grad_clip=2


exp_name="llama2-7B_img224-patch16_${llama_type}_bsz512-5120_lr${lr}_warm${warmup}_clip${grad_clip}_X_v20_finetune"

# srun -p llmeval2 --gres=gpu:8 --cpus-per-task 10 -n8 \
# --ntasks-per-node=8 --quotatype=spot \
srun python -u main_finetune.py \
--epochs ${epochs} \
--warmup_epochs ${warmup} \
--datasets image audio video point rgbd rgbn imu fmri \
--checkpointing \
--max_words 2048 \
--batch_size 4 --accum_iter ${accum} \
--model_parallel_size 1 \
--data_parallel sdp \
--save_consolidated \
--llama_type ${llama_type} \
--init_from "multimodal_llama2_7B/llama2-7B_img224-patch16_llama_clip_resampler_moe_bsz512-5120_lr2e-5_warm2000_clip2_X_v20/epoch_0_iter_000446000/" \
--auto_resume \
--weight_decay 0.0 --output_dir onellm_ckpt/"$exp_name" \
--lr ${lr} --min_lr ${minlr} --clip_grad ${grad_clip} \
--save_interval 1 \
2>&1 | tee -a onellm_ckpt/"$exp_name"/output.log
