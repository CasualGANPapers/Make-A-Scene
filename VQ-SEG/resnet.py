import torch.nn as nn
import torch
import math

__all__ = ['ResNet', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.model(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.alphas = [0.1, 0.25*0.01, 0.25*0.1, 0.25*0.2, 0.25*0.02]
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.channels = [64, 256, 512, 2048, 2048]
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3]),
            NetLinLayer(self.channels[4]),
        ])

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
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

    def forward(self, x):  # 224x224
        features = []
        x = self.conv1(x)  # 112x112
        features.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56x56 ignore

        x = self.layer1(x)  # 56x56
        features.append(x)
        x = self.layer2(x)  # 28x28
        features.append(x)
        x = self.layer3(x)  # 14x14 ignore
        x = self.layer4(x)  # 7x7
        features.append(x)

        x = self.avgpool(x)  # 1x1
        features.append(x)
        
        if not self.include_top:
            return features
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def norm_tensor(x):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + 1e-10)

    @staticmethod
    def spatial_average(x):
        return x.mean([2, 3], keepdim=True)

    def face_loss(self, x, x_rec):
        """
        Takes in original image and reconstructed image and feeds it through face network and takes the difference
        between the different resolutions and scales by alpha_{i}.
        Normalizing the features and applying spatial resolution was taken from LPIPS and wasn't mentioned in the paper.
        """
        true_features = self(x)
        rec_features = self(x_rec)
        # diffs = [a*torch.abs(self.norm_tensor(tf) - self.norm_tensor(rf)) for a, tf, rf in zip(self.alphas, true_features, rec_features)]
        diffs = [a*torch.abs(tf - rf) for a, tf, rf in zip(self.alphas, true_features, rec_features)]
        return sum([net_layer(self.spatial_average(diff)) for net_layer, diff in zip(self.lins, diffs)])


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = resnet50()
    x = torch.randn(1, 3, 500, 500)
    x_rec = torch.randn(1, 3, 500, 500)
    print(model.face_loss(x, x_rec))
