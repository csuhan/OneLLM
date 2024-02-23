## OneLLM: One Framework to Align All Modalities with Language

[[Project Page](https://onellm.csuhan.com)] [[Paper](https://arxiv.org/abs/2312.03700)] [[HF Demo🤗](https://huggingface.co/spaces/csuhan/OneLLM)] [[Modelscope Demo🤖](https://modelscope.cn/studios/csuhan/OneLLM)] [[Model🤗](https://huggingface.co/csuhan/OneLLM-7B)]

## News

- **2023.12.01** Release model weights and inference code.🎉

## Contents

- [Install](#install)
- [Models](#models)
- [Demo](#demo)

### TODO

- [ ] Data
- [ ] Evaluation
- [ ] Training

### Initial Sagemaker setup

To install PointNet in a SageMaker Studio Space (either JupyterLab or Code Editor) we need to install the Cuda Toolkit...

```bash
sudo apt-get update
sudo apt-get install cuda-toolkit-11-7 -y
export CUDA_HOME=/usr/local/cuda-11.7
```

Though the Cuda Toolkit is installed to ephemeral storage it isn't needed once the initial setup is complete.

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/Prevayl/OneLLM

cd OneLLM
```

2. Install packages.

```bash
python -m venv ./.venv
source .venv/bin/activate

pip install -r requirements.txt
```

Install pointnet
```bash
cd model/lib/pointnet2
python setup.py install
cd ../../..
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

```bash
python app.py
```

<!-- ### Evaluation -->

<!-- ### Training -->

## Citation

```
@article{han2023onellm,
  title={OneLLM: One Framework to Align All Modalities with Language},
  author={Han, Jiaming and Gong, Kaixiong and Zhang, Yiyuan and Wang, Jiaqi and Zhang, Kaipeng and Lin, Dahua and Qiao, Yu and Gao, Peng and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2312.03700},
  year={2023}
}
```

## Acknowledgement

[LLaMA](https://github.com/facebookresearch/llama), [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), [Meta-Transformer](https://github.com/invictus717/MetaTransformer), [ChatBridge](https://github.com/joez17/ChatBridge)

## License
This project is developed based on Llama 2, please refer to the [LLAMA 2 Community License](LICENSE_llama2).
