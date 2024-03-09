## OneLLM: One Framework to Align All Modalities with Language

[[Project Page](https://onellm.csuhan.com)] [[Paper](https://arxiv.org/abs/2312.03700)] [[HF Demo🤗](https://huggingface.co/spaces/csuhan/OneLLM)] [[Modelscope Demo🤖](https://modelscope.cn/studios/csuhan/OneLLM)] [[Model🤗](https://huggingface.co/csuhan/OneLLM-7B)]

## News

- **2024.02.27** OneLLM is accepted by **CVPR 2024**!🎉
- **2023.12.01** Release model weights and inference code.

## Contents

- [Install](#install)
- [Models](#models)
- [Demo](#demo)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/csuhan/OneLLM

cd OneLLM
```

2. Install packages.

```bash
conda create -n onellm python=3.9 -y
conda activate onellm

pip install -r requirements.txt

# install pointnet
cd model/lib/pointnet2
python setup.py install
```

3. Install Apex. (Optional)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### Models

We provide a preview model on the Hugging Face at: [csuhan/OneLLM-7B](https://huggingface.co/csuhan/OneLLM-7B).

### Demo

**Huggingface Demo:** [csuhan/OneLLM](https://huggingface.co/spaces/csuhan/OneLLM).

**Local Demo:** Assume you have downloaded the weights to ${WEIGHTS_DIR}. Then run the following command to start a gradio demo locally.

```bash
python demos/multi_turn_mm.py --gpu_ids 0 --tokenizer_path config/llama2/tokenizer.model --llama_config config/llama2/7B.json --pretrained_path ${WEIGHTS_DIR}/consolidated.00-of-01.pth
```

**CLI Demo:**
```bash
python demos/cli.py --image_path ${IMAGE_PATH} --gpu_ids 0 --tokenizer_path config/llama2/tokenizer.model --llama_config config/llama2/7B.json --pretrained_path ${WEIGHTS_DIR}/consolidated.00-of-01.pth
```

### Data

Please check [Data.md](docs/Data.md) for more detail.

### Evaluation

Please check [Evaluation.md](docs/Evaluation.md) for more detail.

### Training

#### Image-Text Pretraining

**Single Node 8-GPU Training**: [exps/image_text_pretrain_8gpu.sh]()
<details><summary>Show More</summary>

```bash
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
--warmup_iters 2000 --lr_decay_iters 200000 --lr 5e-5 --min_lr 5e-6 --clip_grad 2 \
--save_freq 1000 \
2>&1 | tee -a ${OUTPUT_DIR}/output.log
```
</details>

**Multi Node SLURM Training**: [exps/image_text_pretrain_slurm.sh]()
<details><summary>Show More</summary>

```bash
#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH -n 16
#SBATCH -N 2
#SBATCH --cpus-per-task=16

srun python -u main_pretrain.py \
--epochs 1 --dataset image \
--batch_size 40 --accum_iter 8 \
--model_parallel_size 1 \
--data_parallel sdp \
--save_consolidated \
--llama_type onellm \
--llama_ckpt_dir ${LLAMA_7B_PATH} \
--llama_config config/llama2/7B.json \
--tokenizer_path config/llama2/tokenizer.model \
--auto_resume \
--weight_decay 0.1 --output_dir ${OUTPUT_DIR} \
--warmup_iters 2000 --lr_decay_iters 200000 --lr 5e-5 --min_lr 5e-6 --clip_grad 2 \
--save_freq 1000 \
2>&1 | tee -a ${OUTPUT_DIR}/output.log
```
</details>

#### Multimodal-Text Pretraining

**Stage II Pretraining**: Assume we have the pretrained `${IMAGE_TEXT_MODEL}`, run [exps/multimodal_text_pretrain_stage2.sh]() for video-audio-point-text pretraining.

**Stage III Pretraining**: Assume we have the pretrained `${STAGE2_MODEL}`, run [exps/multimodal_text_pretrain_stage3.sh]() for depth-normal-imu-fmri-text pretraining.

#### Instruction Tuning

Assume we have the pretrained `${STAGE3_MODEL}`, run [exps/multimodal_text_finetune.sh]() for multimodal instruction tuning.

## Citation

```
@InProceedings{han2023onellm,
  title={OneLLM: One Framework to Align All Modalities with Language},
  author={Han, Jiaming and Gong, Kaixiong and Zhang, Yiyuan and Wang, Jiaqi and Zhang, Kaipeng and Lin, Dahua and Qiao, Yu and Gao, Peng and Yue, Xiangyu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Acknowledgement

[LLaMA](https://github.com/facebookresearch/llama), [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), [Meta-Transformer](https://github.com/invictus717/MetaTransformer), [ChatBridge](https://github.com/joez17/ChatBridge)

## License
This project is developed based on Llama 2, please refer to the [LLAMA 2 Community License](LICENSE_llama2).
