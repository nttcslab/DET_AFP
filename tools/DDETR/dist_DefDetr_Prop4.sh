#!/usr/bin/env bash

CONFIG=configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_Prop_FT4.py
CHECK=checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth
WORKDIR=./Results/DDETR/Prop_FT4_AFP

GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train_sbt_from_check.py $CONFIG  --launcher pytorch ${@:3}  --checkpointmodel $CHECK  --work-dir $WORKDIR


