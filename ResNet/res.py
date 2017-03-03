import torch.nn as nn
import math


# basic convolution layer
def convolution(input_plane, output_plane, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(input_plane, output_plane, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Basic block of ResNet
class BasicBlock(nn.Module):
    def __init__(self, input_plane, output_plane, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = convolution(input_plane, output_plane, stride)
        self.bn1 = nn.BatchNorm2d(output_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convolution(output_plane, output_plane)
        self.bn2 = nn.BatchNorm2d(output_plane)
        self.short_cut = shortcut
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if input_plane is not same with the output_plane
        # we need to do down_sampling the residual value.
        if self.short_cut is not None:
            residual = self.short_cut(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block_type, num_layer, num_classes=10):
        super(ResNet, self).__init__()
        self.input_plane = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_type, 16, 2 * num_layer[0])
        self.layer2 = self._make_layer(block_type, 32, 2 * num_layer[1], stride=2)
        self.layer3 = self._make_layer(block_type, 64, 2 * num_layer[2], stride=2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # make the shortcut layer
    def _make_layer(self, block, planes, num_block, stride=1):
        shortcut = None
        if stride != 1:
            shortcut = nn.Sequential(
                nn.Conv2d(self.input_plane, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.input_plane, planes, stride, shortcut))
        self.input_plane = planes
        for i in range(1, num_block):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet_56():
    model = ResNet(BasicBlock, [9, 9, 9])
    return model


def resnet_110():
    model = ResNet(BasicBlock, [18, 18, 18])
    return model

