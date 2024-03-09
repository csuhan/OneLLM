## Evaluation

**Annotation Download:** Download the annotations of evaluation datasets from: [csuhan/OneLLM_Eval](https://huggingface.co/datasets/csuhan/OneLLM_Eval), and put it under `datasets/Eval`.

### Image-Text Evaluation

#### COCO Caption

- Download [COCO2014 Val](http://images.cocodataset.org/zips/val2014.zip) and put it under `datasets/InstructionTuning/image/coco/val2014`
- Fill `pretrained_path` in [eval/image_cap_cococap.py]() and run: `python eval/image_cap_cococap.py`
- Install `https://github.com/salaniz/pycocoevalcap`
- Evaluate with [eval/caption_eval.py]()

#### MMVet

- Download MMVet from [mm-vet.zip](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) and put it under `datasets/Eval/image/mm-vet`
- Fill `pretrained_path` in [eval/image_bench_mmvet.py]() and run: `python eval/image_bench_mmvet.py`
- Submit the result file to [Oneline Eval Server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)

### Video-Text Evaluation

#### MSVD QA
- Download [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) video clips from [this link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar) and put it under `datasets/Eval/video/MSVD/YouTubeClips`
- Fill `pretrained_path` in [eval/video_qa_msvd.py]() and run: `python eval/video_qa_msvd.py`.

### Audio-Text Evaluation

#### Clotho Caption
- Download [Clothov2](https://zenodo.org/records/4783391) evaluation set from [this link](https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z?download=1) and put it under `datasets/Eval/audio/clothov2/evaluation`
- Fill `pretrained_path` in [eval/audio_cap_clothov2.py]() and run: `python eval/audio_cap_clothov2.py`.
- Evaluate with [eval/caption_eval.py]().

### Point-Text Evaluation

#### PointLLM Caption
- Download PointLLM data from [this link](https://huggingface.co/datasets/RunsenXu/PointLLM)
- Fill `pretrained_path` in [eval/point_cap_pointllm.py]() and run: `python eval/point_cap_pointllm.py`.
- Evaluate with [eval/caption_eval.py](). The annotation file is at [datasets/Eval/point/pointllm_test_cococap.json]()

### Depth/Normal-Text Evaluation

TODO

### IMU-Text Evaluation

#### Ego4D IMU Caption

- Download Ego4D IMU data. Please refer to [docs/Data.md]().
- Fill `IMU_PATH` and `pretrained_path` in [eval/imu_cap_ego4d.py]() and run: `python eval/imu_cap_ego4d.py`.
- Evaluate with [eval/caption_eval.py]()

### fMRI-Text Evaluation

#### NSD Caption
- Download NSD data. Please refer to [docs/Data.md]().
- Fill `pretrained_path` in [eval/fmri_cap_nsd.py]() and run: `python eval/fmri_cap_nsd.py`.
- Evaluate with [eval/caption_eval.py]()