from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# COCO Caption 
annotation_file = 'datasets/Eval/image/coco_cap/coco_karpathy_val_gt.json'
results_file = 'eval/results/eval_cococap.json'

# Nocaps Caption
# annotation_file = 'datasets/Eval/image/nocaps/nocaps_val_4500_captions.json'
# results_file = 'eval/results/eval_nocaps.json'

# Clotho Caption
# annotation_file = 'datasets/Eval/audio/clothov2/eval_cococap_ann.json'
# results_file = 'eval/results/clotho_13B.json'

# AVSD
# annotation_file = 'datasets/Eval/video/AVSD/test_set4DSTC7-AVSD_cococap.json'
# results_file = 'eval/results/eval_avsd.json'

# VATEX
# annotation_file = 'datasets/Eval/video/vatex/vatex_cococap.json'
# results_file = 'eval/results/eval_vatex.json'

# VALOR32K
# annotation_file = 'datasets/Eval/video/valor32k/test_ann_cococap.json'
# results_file = 'eval/results/eval_videocap_valor.json'

# fMRI Caption
# annotation_file = "datasets/Eval/fmri/fmri_eval_cococap.json"
# results_file = "eval/results/fmricap.json"

# PointLLM Caption
# annotation_file = "datasets/Eval/point/pointllm/pointllm_test_cococap.json"
# results_file = "eval/results/eval_pointllm_cap.json"

# IMU Caption
# annotation_file = 'datasets/Eval/imu/imu_2000_cococap.json'
# results_file = 'eval/results/imucap.json'

# create coco object and coco_result object
coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')
