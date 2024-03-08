#!/bin/bash

STAGE3_MODEL=""
OUTPUT_DIR=""

torchrun --nproc_per_node=8 main_finetune.py \
--epochs 1 --warmup_epochs 0.05 \
--datasets image audio video point rgbd rgbn imu fmri \
--max_words 2048 --batch_size 4 --accum_iter 4 \
--model_parallel_size 1 \
--data_parallel sdp \
--checkpointing --save_consolidated \
--llama_type onellm \
--init_from ${STAGE3_MODEL} \
--auto_resume \
--weight_decay 0.0 --output_dir ${OUTPUT_DIR} \
--lr 2e-5 --min_lr 0.0 --clip_grad 2 \
--save_interval 1 \
2>&1 | tee -a ${OUTPUT_DIR}/output.log
