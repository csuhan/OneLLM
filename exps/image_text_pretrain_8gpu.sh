#!/bin/bash

LLAMA_7B_PATH=""
OUTPUT_DIR=""

torchrun --nproc_per_node=8 main_pretrain.py \
--epochs 1 --dataset image \
--batch_size 40 --accum_iter 16 \
--model_parallel_size 1 \
--data_parallel sdp \
--save_consolidated \
--llama_type onellm \
--llama_ckpt_dir ${LLAMA_7B_PATH} \
--llama_config config/llama2/7B.json \
--tokenizer_path config/llama2/tokenizer.model \
--auto_resume \
--weight_decay 0.1 --output_dir ${OUTPUT_DIR} \
--warmup_iters 2000 --lr_decay_iters 400000 --lr 5e-5 --min_lr 5e-6 --clip_grad 2 \
--save_freq 1000 \
2>&1 | tee -a ${OUTPUT_DIR}/output.log