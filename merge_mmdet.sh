wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.19.0.zip
unzip ./v2.19.0.zip
cp -r -n ./mmdetection-2.19.0/* ./
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
mkdir checkpoints
mv ./faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth ./checkpoints/