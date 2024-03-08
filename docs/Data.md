## Data

### Data Format
Here we give an overview of data format. For details, please check the data loading code: []() and [data/fintune_dataset.py]()

#### Pretraining Data
All the data except IMU are organized in `.csv` format. Each `.csv` has two columns: `caption` and `url`. `\t` is used as the delimiter. For example,
```
caption url
Woman receiving a foot massage at health spa Stock Photo     cluster_p_ssd:s3://laion400m_mmg_ssd/29347/293477138.jpg
Long injury list troubles Paul Hart as Portsmouth search for some Cup form      cluster_p_ssd:s3://laion400m_mmg_ssd/43069/430692001.jpg
... ...
```

#### Instruction Tuning Data
All finetuning data are converted into multi-turn conversation format. The `.json` file contains a list of training samples, where each sample contains the following keys: `id`, `image` and `conversations`. For example,
```
{'id': '000000033471', 'image': 'InstructionTuning/image/coco/train2017/000000033471.jpg', 'conversations': [{'from': 'human', 'value': 'What are the colors of the bus in the image?'}, {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}]}
```


### Download Links

| Stage    | Pretraining    |          | Instruction Tuning         |          |
|----------|-------------|----------|--------------------|----------|
| Modality | Dataset     | Download | Dataset            | Download |
| Image    | [LAION-400M](https://laion.ai/blog/laion-400-open-dataset)  | [link](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md)     | LLaVA-mix665K      | [link](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning)     |
|          | LAION-COCO  | [link](https://laion.ai/blog/laion-coco)     | COCO Caption       | [link](https://cocodataset.org/#download)     |
| Video    | WebVid-2.5M | [link](https://github.com/m-bain/webvid)     | [MSRVTT Caption](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)     | [link](https://www.mediafire.com/folder/h14iarbs62e7p/shared)     |
|          |             |          | MSRVTT-QA          | [link](https://github.com/xudejing/video-question-answering)     |
|          |             |          | [Video Conversation](https://github.com/joez17/ChatBridge/blob/main/custom_datasets/valor_data/DATASET.md#download-multis) | [link](https://drive.google.com/file/d/1C7k8flfITJ1GxMwFSvEmBFGyevDZl1ke/view?usp=drive_link)     |
| Audio    | [WavCaps](https://github.com/XinhaoMei/WavCaps)     |  [link](https://huggingface.co/datasets/cvssp/WavCaps)    | [AudioCaps](https://audiocaps.github.io/)          | [link](https://github.com/cdjkim/audiocaps)     |
|          |             |          | [Audio Conversation](https://github.com/joez17/ChatBridge/blob/main/custom_datasets/valor_data/DATASET.md#download-multis) | [link](https://drive.google.com/file/d/1C7k8flfITJ1GxMwFSvEmBFGyevDZl1ke/view?usp=drive_link)     |
| Point    | [Cap3D](https://github.com/crockwell/Cap3D)       | [link](https://huggingface.co/datasets/RunsenXu/PointLLM/tree/main)     | [Point Conversation](https://github.com/OpenRobotLab/PointLLM) | [link](https://huggingface.co/datasets/RunsenXu/PointLLM)     |
| Depth    | CC3M        | [link](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)     | LLaVA-150K         | [link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)     |
| Normal   | CC3M        | [link](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)     | LLaVA-150K         | [link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)     |
| IMU      | Ego4D       | [link](https://ego4d-data.org/docs/data/imu/)     | Ego4D              | [link](https://ego4d-data.org/docs/data/imu/)     |
| fMRI     | [NSD](https://naturalscenesdataset.org)         | [link](https://huggingface.co/datasets/pscotti/naturalscenesdataset)     | [NSD](https://naturalscenesdataset.org)                | [link](https://huggingface.co/datasets/pscotti/naturalscenesdataset)     |

**Notes**
- The depth/normal map are generated from [CC3M](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) and 50K random-subset of LLaVA-150K using a pretrained [DPT](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#run-our-models-on-your-own-image).
- The [IMU data](https://ego4d-data.org/docs/data/imu/) is preprocessed with [this script](https://github.com/facebookresearch/imu2clip/blob/main/dataset/ego4d/preprocessing_scripts/extract_imu.py).


### Instruction Tuning Data

**Annotation Download:** Please download the annotation from [this link](https://huggingface.co/datasets/csuhan/OneLLM_InstructionTuning) and put them under `datasets/InstructionTuning`.

Then download original datasets from the above table and put them under corresponding folders. The file structure should be:

```
datasets
└── InstructionTuning
    ├── audio
    │   ├── audioset2
    │   ├── audiocap_train.json
    │   ├── audiocap_val.json
    │   └── audio_conversation.json
    ├── depth_normal
    │   ├── depth
    │   ├── normal
    │   ├── llava_instruct_50k_depth.json
    │   └── llava_instruct_50k_normal.json
    ├── fmri
    │   ├── NSD
    │   └── fmri_fixed_train.json
    ├── image
    │   ├── coco
    │   ├── gqa
    │   ├── ocr_vqa
    │   ├── vg
    │   ├── cococap_train.json
    │   ├── llava_v1_5_mix665k_image.json
    │   └── llava_v1_5_mix665k_text.json
    ├── imu
    │   ├── ego4d
    │   └── imu_fixed_50k.json
    ├── point
    │   ├── pointllm/8192_npy
    │   └── pointllm_70k.json
    └── video
        ├── msr-vtt/MSR-VTT
        ├── msrvtt_cap_test.json
        ├── msrvtt_cap_trainval.json
        ├── msrvtt_vqa_test.json
        ├── msrvtt_vqa_train.json
        ├── msrvtt_vqa_val.json
        ├── video_complex_reasoning_10k.json
        ├── video_conversation_10k.json
        └── video_detail_10k.json
```