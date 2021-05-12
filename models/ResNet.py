import torch.nn as nn
import torch
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.fea_dim = 64 * block.expansion  # the dimension of the output of the penultimate layer, which is the dimension of the feature

        # Initialize the weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Methods in the source code of the paper
                '''
                # This part is the default method in the network source code file
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''
            elif isinstance(m, nn.BatchNorm2d):
                # The following is also the default method in the network source code file, which is a little different from the source code of the paper, but it should not be a big problem, so we don’t change it.
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # If the dimensions of the residual and the output of the convolution in the block do not match, the residual needs to be down-sampled, which is achieved by 1*1 convolution
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, feat_s=None):
        '''

        :param x:
        :param is_adain: Whether to output the feature generated by the penultimate layer
        :param feat_s: Feature input directly to the last layer
        :return:
        '''
        if not feat_s is None:  # If there is feat_s, then directly input feat_s to the last layer and return the output result
            x = self.avgpool(feat_s)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        # if feat_s is None
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        fea = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if is_adain:  # If is_adain is true, return the feature generated by the penultimate layer incidentally
            return fea, x
        else:
            return x


def resnet8():
    model = ResNet(8, 10, bottleneck=False)  # Number of parameters: 78042  cifar10 student
    return model


def resnet14():
    model = ResNet(14, 10, bottleneck=False)  # Number of parameters: 175258 cifar10 student
    return model


def resnet26(file="models/resnet26_cifar10_9147.pth"):
    model = ResNet(26, 10, bottleneck=False)  # Number of parameters: 369690 cifar10 teacher
    model.load_state_dict(torch.load(file))
    return model



if __name__ == '__main__':
    model = resnet26()
    model.cuda()
    model1 = ResNet(18, 10, bottleneck=False)
    model2 = ResNet(18, 10, bottleneck=True)
    print(sum(p.numel() for p in model1.parameters()), sum(p.numel() for p in model2.parameters()))
