import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *
from networks.resnet_local_grad import resnet50_local_grad
__all__ = ['resnet50_gradient']

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

class ResnetGrad(nn.Module):
    def __init__(self, model_A_path, pretrainedA, pretrainedB, **kwagrs):
        super(ResnetGrad,self).__init__()
        self.model_A = resnet50_local_grad(pretrained=pretrainedA, num_classes=1)
        self.model_A.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.model_A.load_state_dict(torch.load(model_A_path, map_location='cpu'), strict=True)
        self.model_A.eval()
        for param in self.model_A.parameters():
            param.requires_grad = False
    
        self.model_B = resnet50_local_grad(pretrained=pretrainedB, num_classes=1)
        self.model_B.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    
    def forward(self,x):
        input_tensor = x.clone()
        input_tensor.requires_grad_()
        output_A = self.model_A(input_tensor)
        #loss = output_A.sum()
        #loss.backward()
        loss = output_A.sum(dim=1) 
        loss.backward(torch.ones_like(loss))
        
        gradients = input_tensor.grad
        mask = (F.relu(gradients) > 0).float()
        
        x = x*mask
        x = self.model_B(x)
        return x


def resnet50_gradient(model_A_path, pretrainedA=True,pretrainedB=False,**kwargs):
    model = ResnetGrad(model_A_path=model_A_path,
                       pretrainedA=pretrainedA, 
                       pretrainedB=pretrainedB)
    return model
    

if __name__ == '__main__':
    from options.train_options import TrainOptions
    from util import get_model
    import torch
    opt = TrainOptions().parse()
    opt.detect_method = 'local_grad'
    model = get_model(opt)
    #model = resnet50_local_grad_new(pretrained=True)

    intens = torch.rand(2,3,224,224)
    out = model(intens)    

    
    
    
    
    
    
    