# Object-Detection-level2-9
P-Stage level-2 object detection competition. (이미지 내 쓰레기 객체 감지 및 분류)<br>
9조 (하나둘셋Net()) Solution

# Archive contents
```
baseline/
├── mmdetection/
│ ├── cascade_rcnn_swin/
| | ├── cascade_rcnn_swin_large.py
| | ├── cascade_rcnn_swin.py
| | ├── cascade_rcnn.py
| | ├── dataset.py
| | ├── runtime.py
| | └── scheduler.py
│ ├── faster_rcnn/
| | ├── faster_rcnn_r152_fpn_dconv_c3-c5_1x_coco.py
| | ├── faster_rcnn_r152_fpn_1x_coco.py
| | ├── faster_rcnn_r50_fpn_1x_coco.py
| | ├── faster_rcnn_r50_fpn.py
| | ├── dataset.py
| | ├── default_runtime.py
| | └── schedule_1x.py
| └── inference.ipynb
├── ensemble/
│ └── Ensemble.ipynb
├── YOLOv5/
│ ├── data/
| | └── coco_trash.yaml
│ ├── runs/
| | ├── hyp.yaml
| | └── opt.yaml
└── inference.ipynb
eda/
├── train.py
├── inference.py
├── dataset.py
├── evaluation.py
├── loss.py
├── model.py
└── model
utils/
|  └── inference_checker/
|    └── main.py
|  └── trainset_check/
└──  └── main.py
```
- `input/data/eval`: evaluation dataset
- `input/data/train`: train dataset
- `code/train.py`: main script to start training
- `code/inference.py`: evaluation of trained model
- `code/dataset.py`: custom data loader for dataset
- `code/evaluation.py`: function to score, matric
- `code/loss.py`: contains loss functions
- `code/model.py`: contains custom or pre-trained model
- `model/`: trained models are saved here like(exp1, exp2,...) <br>


# Requirements
- Linux version 4.4.0-59-generic
- Python >= 3.8.5
- PyTorch >= 1.7.1
- conda >= 4.9.2
- tensorboard >= 2.4.1

# Reference
`git clone https://github.com/open-mmlab/mmdetection.git` : install mmdetection library <br>
`git clone https://github.com/ultralytics/yolov5.git` : install YOLOv5 library. <br>

### Hadware
- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Tesla V100-SXM2-32GB

# Training
## mmdetecction
`cd mmdetection` <br>
`python tools/train.py configs` <br>

## YOLOv5
`cd YOLOv5` <br>
`python train.py --img 1024 --batch 4 --epochs 50 --data coco128.yaml --weights yolov5x6.pt`


# Inference
```python inference.py --model_dir {'MODEL_PATH'}```
<br>ex. <br>
`python inference.py --model_dir "./models/exp1"`
