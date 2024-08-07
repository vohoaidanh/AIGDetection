import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *

from networks.local_grad import gradient_filter
__all__ = ['build_model']

class ConditionalAvgPool2d(nn.Module):
    def __init__(self):
        super(ConditionalAvgPool2d, self).__init__()
    
    def forward(self, x):
        if x.dim() == 4:
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            avgpool = nn.AdaptiveAvgPool1d(1)
        x = avgpool(x)
        
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
class PreProcess(nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()
        self.process = gradient_filter
    
    def forward(self,x):
        return self.process(x)
    

def get_backbone(backbone_name, layer_to_extract=-1, pretrained=True):
    if backbone_name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif backbone_name == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif backbone_name == "vgg16":
        model = models.vgg16(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    backbone = nn.Sequential(*list(model.children())[:layer_to_extract])
    output_shape = backbone(torch.rand(1,3,224,224)).shape

    return backbone, output_shape


def build_model(backbone_name, num_classes, layer_to_extract):
    backbone, output_shape = get_backbone(backbone_name, layer_to_extract)
    print(f'output_shape of bacbone is {output_shape}')
    in_features = output_shape[1]
    preprocess = PreProcess()
    connection = ConditionalAvgPool2d()
    head = ClassificationHead(in_features, num_classes)
    
    # Create the final model
    model = nn.Sequential(
        preprocess,
        backbone,
        connection,  # Global average pooling
        nn.Flatten(),  # Ensure the output is flattened before passing to the head
        head
    )
    
    return model



if __name__ == '__main__':
    #from options.train_options import TrainOptions
    #from util import get_model
    #import torch
    #opt = TrainOptions().parse()
    #opt.detect_method = 'local_grad'
    #model = get_model(opt)
    #model = resnet50_local_grad_new(pretrained=True)

    #intens = torch.rand(2,3,224,224)
    #out = model(intens)    
    
    # Example usage
    backbone_name = "resnet50"  # Choose from "resnet50", "resnet18", "vgg16"
    num_classes = 1  # Number of classes for classification
    layer_to_extract = "layer2"  # For ResNet, choose the layer to extract features from

    model = build_model(backbone_name='vgg16', num_classes=1, layer_to_extract=-1)
    print(model)
    intens = torch.rand(2,3,224,224)
    out = model(intens)    
    
    
    model = get_backbone('resnet50')
    
    avgpool = nn.AdaptiveAvgPool1d(1)
    avgpool(torch.rand(2,22,13)).shape
    
    connection = ConditionalAvgPool2d()
    out = connection(torch.rand(3,4))
    out.shape
    nn.Flatten()(out).shape
    connection(torch.rand(3,4,5,6))



