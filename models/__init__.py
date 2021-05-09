from .official_resnet import resnet34T,resnet18S,resnet50T
from .MobileNet import MobileNet
from .ResNet import resnet8, resnet14, resnet26

model_dict = {
    'resnet50T':resnet50T,
    'resnet34T':resnet34T,
    'resnet18S':resnet18S,
    'MobileNet':MobileNet,
    'resnet8':resnet8,
    'resnet14':resnet14,
    'resnet26':resnet26

}
