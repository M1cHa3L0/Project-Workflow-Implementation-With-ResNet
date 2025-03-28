"""
paper and code replicating
model file
"""
import torch
from torch import nn

# 残差块类
class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, indentity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4 # 输出的channel数是进入时的4倍
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU()
        self.indentity_downsample = indentity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.indentity_downsample is not None: # 第一块残差块会高宽减半，减少特征图的尺寸，后续的残差块就不会
            identity = self.indentity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# resnet类
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes): # [残差块数量， 残差块层数， input channels, 输出类别数量] eg. resnet-50: layers-> [3,4,6,3]
        super(ResNet, self).__init__()
        # conv1: change image channel to 64
        self.in_channels = 64 # 进入残差块的channel
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layer(conv2-5)
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2) # 最后的intermediateput_channel=512*4
        # average pool and fully connect
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x



    # resnet layer function
    def _make_layer(self, block, num_residual_block, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(intermediate_channels*4)
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride)) # intermediate_channel = 64
        self.in_channels = intermediate_channels*4 # 256

        for i in range(num_residual_block-1):
            # intermediate_channel是64, 但是在最后一次conv后的intermediate_channel会x4
            layers.append(block(self.in_channels, intermediate_channels)) # 256 -> 64 -> 256
        
        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3,8,36,3], img_channels, num_classes)


def test():
    net = ResNet50()
    x = torch.rand(2,3,224,224)
    y = net(x)
    print(y.shape)


test()