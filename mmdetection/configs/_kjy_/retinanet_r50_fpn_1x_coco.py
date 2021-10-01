_base_ = [
    '../_custom_/models/retinanet_r50_fpn.py',
    '../_custom_/datasets/coco_detection.py',
    '../_custom_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
