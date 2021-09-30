# MMDetection 팀 활동 

# Archive

```
configs/
├── _base_/
| ├── datasets/
| ├── models/
| ├── schedules/
| └── default_runtime.py
├── _custom_/
| ├── datasets/
| | ├── coco_detection.py
| ├── models/
| | ├── cascade_rcnn_r50_fpn.py
| ├──schedules/
| └──└── schedule_2x.py
├── _kjy_/
| └── cascade_rcnn_r50_fpn_1x_coco.py
```

# 폴더 설명
- ```_custom_``` : ```_base_``` 폴더에 있는 파일 뿐 아니라 configs에 들어있는 여러 파일들을 cumstom하여 수정한 파일을 저장하는 폴더입니다. ```datasets```, ```models```, ```schedules``` 폴더가 들어있습니다. 팀원들이 어떤 custom을 했는지에 따라 ```datasets```, ```models```, ```schedules``` 폴더에 맞게 저장합니다. 예를 들면, ```coco_detection.py``` 파일은 기존에 있던 파일에서 클래스 이름(General trash, ...)을 새로 설정 해주었고 ```datasets/``` 폴더에 저장하였습니다. ```cascade_rcnn_r50_fpn.py```파일은 모델의 num_classes = 10으로 수정하여 ```models/``` 폴더에 저장하였습니다.

- ```_kjy_``` : 개인 폴더를 생성하여 _custom_ 폴더에 들어있는 파일을 ```cascade_rcnn_r50_fpn_1x_coco.py``` 파일에 다음과 같은 형식으로 저장합니다.
```
_base_ = [
    '../_custom_/models/cascade_rcnn_r50_fpn.py',
    '../_custom_/datasets/coco_detection.py',
    '../_custom_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
```

# 서버에 깃 클론하는 방법

1. 노션 링크에 들어가 ```서버에 깃 클론하는 방법``` pdf를 참조하여 15번까지 따라합니다.  https://www.notion.so/MM-Detection-f79a79b5d89c41f4ba6b7f387bdad62b
2. ``` cd ./detection ``` detection 폴더에 들어갑니다.
3. ``` git clone git@github.com:boostcampaitech2/object-detection-level2-cv-09.git ```
4. ``` cd object-detection-level2-cv-09/mmdetection ```
5. ``` python tools/train.py configs/_kjy_/cascade_rcnn_r50_fpn_1x_coco.py ``` 를 실행해보고, loss 값이 찍히면 성공적으로 설치가 완료된 것입니다.


# config 작명 방법
이름이 ```faster_rcnn_r50_fpn_1x_coco```와 같이 꽤 긴 것을 볼 수 있다. 많은 정보를 담고 있는데, 일반적인 형식은 다음과 같다.

```{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}```
```{중괄호}```는 필수, ```[대괄호]```는 선택이다.

- ```{model}```: faster_rcnn와 같은 모델 이름이다.
- ```{model setting}```: 일부 모델에 대한 세부 설정인데 ```htc```의 경우 ```without_semantic```, ```reppoints```의 경우 ```moment``` 등이다.
- ```{backbone}```: 모델의 전신이 되는 기본 모델로 ```r50(ResNet-50), x101(ResNeXt-101)``` 등이다.
- ```{neck}```: 모델의 neck에 해당하는 부분을 정하는 것으로 ```fpn, pafpn, nasfpn, c4``` 등이 있다.
- ```{norm_setting}```: 기본값은 bn으로 batch normalization이며 생략이 가능하다. ```gn```은 Group Normalization, ```syncbn```은 Synchronized BN, ```gn-head``` 및 ```gn-neck```은 GN을 head 또는 neck에만 적용, ```gn-all```은 모델의 전체(backbone, nect, head)에다가 GN을 적용한다.
- ```[misc]```: 이모저모를 적자. ```dconv, gcb, attention, albu, mstrain``` 등이다.
- ```[gpu x batch_per_gpu]```: GPU 개수와 GPU 당 sample 개수로 8x2가 기본이다.
- ```{schedule}```: ```1x```는 12epoch, ```2x```는 24epoch이며 8/16번째와 11/22번째 epoch에서 lr이 10분의 1이 된다. 20e는 cascade 모델에서 사용되는 것으로 20epoch으로 10분의 1이 되는 시점은 16/19번째이다.
- ```{dataset}```: 데이터셋을 나타내는 부분으로 ```coco```, ```cityscapes```, ```voc_0712```, ```wider_face``` 등이다. 





# 참고 
https://greeksharifa.github.io/references/2021/09/05/MMDetection02/
