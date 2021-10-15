# Object-Detection-level2-9
P-Stage level-2 object detection competition. (이미지 내 쓰레기 객체 감지 및 분류)<br>
9조 하나둘셋Net() Solution

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
├── eda.ipynb
└── eda-2.ipynb
utils/
|  └── inference_checker/
|    └── main.py
|  └── trainset_check/
└──  └── main.py
```
- `baseline/mmdetection/cascade_rcnn`: config files for Cascade R-CNN model
- `baseline/mmdetection/faser_rcnn`: config files for Faster R-CNN model
- `baseline/mmdetection/inference.ipyb`: inference code for mmdetection library
- `baseline/ensemble/Ensemble.ipynb`: ensemble code 
- `baseline/YOLOv5/data/coco_trash.yaml`: converted Trash dataset to YOLO data form
- `baseline/YOLOv5/runs`: config file of our model 
- `baseline/YOLOv5/inferece.ipynb`: inference code for YOLOv5 library
- `eda/eda.ipynb`: result of EDA
- `eda/eda-2.ipynb`: another result of EDA
- `utils/inference_chcker/main.py`: python program for visualization of result of inference
- `utils/trainset_check/main.py`: python program for visualization of training set <br>


# Requirements
- Linux version 4.4.0-59-generic
- Python >= 3.8.5
- PyTorch >= 1.7.1
- conda >= 4.9.2
- tensorboard >= 2.4.1
### Hardware
- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Tesla V100-SXM2-32GB

# Reference
`git clone https://github.com/open-mmlab/mmdetection.git` : install mmdetection library <br>
`git clone https://github.com/ultralytics/yolov5.git` : install YOLOv5 library. <br>

# Training
**mmdetection** <br>
- mmdetection 라이브러리 clone 후 configs file 실행<br>
**example**<br>
- `cd mmdetection` <br>
- `python tools/train.py baseline/mmdetection/cascade_rcnn/cascade_rcnn_swin_large.py` <br>

**YOLOv5** <br>
**example** <br>
- `cd YOLOv5` <br>
- `python train.py --img 1024 --batch 4 --epochs 50 --data baseline/data/coco_trash.yaml --weights yolov5x6.pt`

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


