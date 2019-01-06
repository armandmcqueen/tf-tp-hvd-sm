# TensorFlow + TensorPack + Horovod + Amazon SageMaker

## Pre-requisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your SageMaker service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so your SageMaker service limit allows you to launch required number of GPU enabled EC2 instances, such as ml.p3.16xlarge. You would need a minimum limit of 2 GPU enabled instances. For the purpose of this setup, a SageMaker service limit of 8 ml.p3.16xlarge instance types is recommended.

3. [Install and configure AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

4. The steps described below require adequate [AWS IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/access.html) permissions.

## Overview

In this project, we are focused on distributed training using [TensorFlow](https://github.com/tensorflow/tensorflow), [TensorPack](https://github.com/tensorpack/tensorpack) and [Horovod](https://eng.uber.com/horovod/) on [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

While all the concepts described here are quite general and are applicable to running any combination of TensorFlow, TensorPack and Horovod based algorithms in Amazon SageMaker, we will make these concepts concrete by focusing on distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example. 

## Prepare S3 Bucket

Next we will stage the data and code in an S3 bucket that will be later used for distributed training in Amazon SageMaker. 

While the idea of using S3 to stage data is quite general, we will make the concept concrete by staging [Coco 2017](http://cocodataset.org/#download) dataset and [COCO-R50FPN-MaskRCNN-Standard](http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-Standard.npz) pre-trained model, so we can do distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example 

1. Customize variables in ```prepare-s3-bucket.sh```
   
   a) S3_BUCKET variable must point to an existing bucket in the same region in which you are planning to do SageMaker training. 
   
   b) STAGE_DIR variable must point to a volume with 50 - 100 GB of available space. 

2. Execute ```nohup ./prepare-s3-bucket.sh &``` to stage data and code needed for distributed training in an S3 bucket.    **You can use the [screen](https://linuxize.com/post/how-to-use-linux-screen/) command as an alternative to using ```nohup``` and ```screen``` appears to work more reliably than ```nohup``` command.**

## Build and Upload Docker Image to ECR

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. It is recommended that the Docker Image be built on an EC2 instance based on [Amazon Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/).

## Create Amazon SageMaker Training Job

1. Upload ```notebook/mask-rcnn.ipynb``` Jupyter Notebook to Amazon SageMaker

2. Customize S3 bucket and any other variables in the notebook.

3. Execute the uploaded notebook to start an Amazon SageMaker training job.

