# SRRL
## Paper
[Knowledge distillation via softmax regression representation learning](https://openreview.net/pdf?id=ZzwDy_wiWv)

Jing Yang, Brais Martinez, Adrian Bulat, Georgios Tzimiropoulos

## Method
<div align="center">
    <img src="overview.png" width="600px"</img> 
</div> 

## Requirements
- Python >= 3.6
- PyTorch >= 1.0.1

## content
- `train_cifar10_teacher.py`: To train the teacher network individually and save its weights. The current highest test set accuracy of ResNet-26 is 0.83.
- `train_cifar10_distillation.py`: The main part of this algorithm. Different from the `train_imagenet_distillation.py` in the source code, this code is on the CIFAR10 dataset.
- `models/ResNet.py`: ResNet networks suitable for the CIFAR10 data set. It already have Resnet-8, ResNet-14 and Resnet-26.
