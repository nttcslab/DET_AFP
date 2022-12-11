#!/usr/bin/env bash

CONFIG=./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_wProp_FT3.py
CHECK=./checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
WORKDIR=./Results/Faster/Prop_FT3_AFP

GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train_sbt_from_check.py $CONFIG  --launcher pytorch ${@:3}  --checkpointmodel $CHECK  --work-dir $WORKDIR

#---------------------------------

