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
### Hadware
- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Tesla V100-SXM2-32GB

# Reference
`git clone https://github.com/open-mmlab/mmdetection.git` : install mmdetection library <br>
`git clone https://github.com/ultralytics/yolov5.git` : install YOLOv5 library. <br>

# Training
 **mmdetection** <br>
mmdetection 라이브러리 clone 후 configs file 실행<br>
**example**<br>
`cd mmdetection` <br>
`python tools/train.py baseline/mmdetection/cascade_rcnn/cascade_rcnn_swin_large.py` <br>

**YOLOv5** <br>
**example** <br>
`cd YOLOv5` <br>
`python train.py --img 1024 --batch 4 --epochs 50 --data baseline/data/coco_trash.yaml --weights yolov5x6.pt`

**Ensemble**<br>
- baseline/ensemble/Ensemble.ipynb 파일 실행
- target folder 생성 후 ensemble 하고자 하는 파일 target1.csv, target2.csv 형식으로 저장
- ipynb 파일 run (Weighted Boxes Fusion)
# Tools
## inference_checker
`cd utils/inference_checker`<br>
`python main.py`<br>
open submission file (command or control + O)
![image](https://user-images.githubusercontent.com/51802825/137460394-b479574c-4340-4fb8-bf19-53f9a4939941.png)

## trainset_check
`cd utils/treainset_check` <br>
`python main.py`<br>
![image](https://user-images.githubusercontent.com/51802825/137460478-603b7610-c7fa-4e83-b632-7ab5335f4499.png)


