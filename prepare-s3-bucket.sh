#!/bin/bash

# Customize S3_BUCKET
S3_BUCKET=

# Customize S3_PREFIX
S3_PREFIX=mask-rcnn/sagemaker/input

# Customize Stage DIR
# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$HOME/stage

if [ -e $STAGE_DIR ]
then
echo "$STAGE_DIR already exists"
exit 1
fi

mkdir -p $STAGE_DIR/train 

wget -O $STAGE_DIR/train/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip $STAGE_DIR/train/train2017.zip  -d $STAGE_DIR/train
rm $STAGE_DIR/train/train2017.zip

wget -O $STAGE_DIR/train/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip $STAGE_DIR/train/val2017.zip -d $STAGE_DIR/train
rm $STAGE_DIR/train/val2017.zip

wget -O $STAGE_DIR/train/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $STAGE_DIR/train/annotations_trainval2017.zip -d $STAGE_DIR/train
rm $STAGE_DIR/train/annotations_trainval2017.zip

mkdir $STAGE_DIR/train/pretrained-models
wget -O $STAGE_DIR/train/pretrained-models/COCO-R50FPN-MaskRCNN-Standard.npz http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-Standard.npz

aws s3 cp --recursive $STAGE_DIR/train s3://$S3_BUCKET/$S3_PREFIX/train
aws s3 cp --recursive code s3://$S3_BUCKET/$S3_PREFIX/code

