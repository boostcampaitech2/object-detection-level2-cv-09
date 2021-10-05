_base_ = [
    '../_custom_/models/retinanet_r50_fpn.py',
    '../_custom_/datasets/coco_detection.py',
    '../_custom_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)