# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights
from networks.local_grad import gradient_filter
#import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
    
class Clf_model(nn.Module):
    
    def __init__(self, original_model, num_classes=1):
        super(Clf_model, self).__init__()
        self.gradient_filter = gradient_filter
        input_dim = 768
        self.features = nn.Sequential(*list(original_model.children())[:-7])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = MLPHead(input_dim=input_dim, output_dim=num_classes)
        
    def forward(self,x):
        x = self.gradient_filter(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def inception_local_grad(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    else:
        inception_model = models.inception_v3(weights=None)

    model = Clf_model(inception_model,**kwargs)
        
    return model


if __name__ == '__main__':
    model = inception_local_grad(pretrained=True, num_classes=1)
    model(torch.rand(1,3,299,299)).shape
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


