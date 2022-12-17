#!/usr/bin/env bash

CONFIG=configs/faster_rcnn/faster_rcnn_X101_fpn_1x_coco_Prop_FT3.py
CHECK=./checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth
WORKDIR=./Results/Faster/Prop_X101_FT3_AFP


GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train_sbt_from_check.py $CONFIG  --launcher pytorch ${@:3}  --checkpointmodel $CHECK  --work-dir $WORKDIR

#---------------------------------
