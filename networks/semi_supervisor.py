# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *
__all__ = ['ResNet', 'resnet_similarity','SimilarityClassifier']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.gradient_layer = gradient_filter  # Instantiate GradientLayer
        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    
    def forward(self, x):
        #NPR = x.clone()
        x = self.gradient_layer(x)
        
        #x = self.conv1(NPR*2.0/3.0)
        #NPR = self.interpolate(NPR, 0.5)
        #NPR = self.gradient_layer(NPR)
        #x = x - NPR
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_local_grad(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict = {k: v for k, v in state_dict.items() if not any(layer in k for layer in ['fc', 'layer3', 'layer4'])}
        model.load_state_dict(new_state_dict,strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class SimilarityClassifier(nn.Module):
    def __init__(self,pretrained):
        print('load SimilarityClassifier')
        super(SimilarityClassifier, self).__init__()
        self.pretrained = pretrained
        self.feature_extractor = resnet50_local_grad(pretrained=self.pretrained)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.classifier = nn.Linear(1, 1)  # Output: 2 classes (0 and 1)
    
    def forward(self, x):
        bz,_,_,_ = x.shape
        x2 = torch.flip(x, dims=[3])
        features1 = self.feature_extractor(x).view(bz,-1)
        features2 = self.feature_extractor(x2).view(bz,-1)
        distance = torch.sqrt(torch.sum((features1 - features2) ** 2, dim=1, keepdim=True))
        output = self.classifier(distance)
        return output
    
class ResnetCenterLoss(nn.Module):
    def __init__(self,pretrained):
        print('load ResnetCenterLoss')
        super(ResnetCenterLoss, self).__init__()
        self.pretrained = pretrained
        self.feature_extractor = resnet50_local_grad(pretrained=self.pretrained)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.classifier = nn.Linear(512, 1)  # Output: 2 classes (0 and 1)
    
    def forward(self, x):
        bz,_,_,_ = x.shape
        features = self.feature_extractor(x).view(bz,-1)
        output = self.classifier(features)
        return features, output


def resnet_similarity(pretrained=False):
    return SimilarityClassifier(pretrained=pretrained)

def resnet_center_loss(pretrained):
    return ResnetCenterLoss(pretrained=pretrained)

if __name__ == '__main__':
    from options.train_options import TrainOptions
    from util import get_model
    import torch
    
    from torchvision import transforms
    from networks.trainer import Trainer

    transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor (HWC -> CHW, values scaled between 0 and 1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    from PIL import Image
    opt = TrainOptions().parse()
    opt.detect_method = 'resnet_center_loss'
    opt.model_path = r'D:/K32/do_an_tot_nghiep/NPR-DeepfakeDetection/weights/Gaussblur-4class-resnet-car-cat-chair-horse2024_06_20_08_12_39_model_eopch_16_last.pth'
    
    
    model = Trainer(opt)
    #model = get_model(opt)

    #feature = resnet_center_loss(pretrained=True)
    img = Image.open(r"D:\dataset\stylegan\bedroom\0_real\14240.png")
    im = transform(img)
    im = im.unsqueeze(0)
    out = model.model(im)
    




    
