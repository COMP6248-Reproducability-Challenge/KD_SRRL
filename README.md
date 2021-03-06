# SRRL
## Target Paper
[Knowledge distillation via softmax regression representation learning](https://openreview.net/pdf?id=ZzwDy_wiWv)

Jing Yang, Brais Martinez, Adrian Bulat, Georgios Tzimiropoulos

[OpenReview](https://openreview.net/forum?id=ZzwDy_wiWv)

[source code](https://github.com/jingyang2017/KD_SRRL)

## Method
<div align="center">
    <img src="overview.png" width="600px"</img> 
</div> 

## Requirements
- Python >= 3.6
- PyTorch >= 1.0.1

## Introduction

We tried to reproduce the experiments on CIFAR-10 using ResNet-8,14,26. We also provided different choices for SR loss, loss weight alpha and beta.

## Main Content
- `train_cifar10_teacher.py`: To train the teacher network individually and save its weights. 
- `train_cifar10_distillation.py`: The main part of this algorithm. Different from the `train_imagenet_distillation.py` in the source code, this code is on the CIFAR10 dataset.
- `models/ResNet.py`: ResNet networks suitable for the CIFAR10 data set. It already have Resnet-8, ResNet-14 and Resnet-26.
- `models/resnet26_cifar10_n.pth`: the weights of ResNet-26 which is the teacher network on the CIFAR10 data set. n means the test accuracy. The current highest test set accuracy is 0.9147.
- `test_cifar10.py`: test the accuracy of the model on the CIFAR10 test set.
