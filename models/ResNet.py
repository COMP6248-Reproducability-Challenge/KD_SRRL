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

        self.fea_dim = 64 * block.expansion  # 倒数第二层输出的维度，也就是feature的维度

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 论文源代码里的方法
                '''
                # 这部分是网络原代码文件里默认的方法
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''
            elif isinstance(m, nn.BatchNorm2d):
                # 以下也是网络原代码文件里默认的方法，和论文源代码有一点点区别，但应该问题不大，所以不改
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # # 如果残差和block中的卷积的输出两者维度不匹配，就需要对残差进行下采样，这是通过1*1卷积来实现的
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
        :param is_adain: 是否输出倒数第二层生成的feature
        :param feat_s: 直接往最后一层输入的feature
        :return:
        '''
        if not feat_s is None:  # 如果feat_s不是None，那么就直接把feat_s输入给最后一层，返回输出结果
            x = self.avgpool(feat_s)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        # feat_s为None的情况
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

        if is_adain:  # 如果is_adain是true，则需要顺带返回倒数第二层生成的feature
            return fea, x
        else:
            return x

    '''
    # 剩下的部分似乎用不上
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x, preReLU=False):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = F.relu(feat3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)

        return [feat1, feat2, feat3], out
    '''


def resnet8():
    model = ResNet(8, 10, bottleneck=False)  # 参数个数：78042  cifar10作为学生
    return model


def resnet14():
    model = ResNet(14, 10, bottleneck=False)  # 参数个数：175258 cifar10作为学生
    return model


def resnet26(file="models/resnet26_cifar10_83.pth"):
    '''
    当老师的料
    :return:
    '''
    model = ResNet(26, 10, bottleneck=False)  # 参数个数：369690 cifar10作为老师
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(file))
    return model


# resnet8 = ResNet(8, 10, bottleneck=False)   # 参数个数：78042  cifar10作为学生
# resnet14 = ResNet(14, 10, bottleneck=False) # 参数个数：175258 cifar10作为学生
# resnet26 = ResNet(26, 10, bottleneck=False) # 参数个数：369690 cifar10作为老师
# 18 和 34 的cifar10版本暂时对不上

if __name__ == '__main__':
    model = resnet26()
    model.cuda()
    # 数网络参数个数
    model1 = ResNet(18, 10, bottleneck=False)
    model2 = ResNet(18, 10, bottleneck=True)
    print(sum(p.numel() for p in model1.parameters()), sum(p.numel() for p in model2.parameters()))
