import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *
import sys
from data import create_dataloader

__all__ = ['ResNet', 'resnet50_local_grad']


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

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layers = layers
        self.feature_outputs = []

        # Register hooks for the specified layers
        self.hook_handles = []
        for layer in layers:
            handle = layer.register_forward_hook(self.hook_fn)
            self.hook_handles.append(handle)

    def hook_fn(self, module, input, output):
        self.feature_outputs.append(output)

    def forward(self, x):
        # Clear previous feature outputs
        self.feature_outputs = []
        
        # Perform forward pass through the model
        _ = self.model(x)
        
        return self.feature_outputs

    def remove_hooks(self):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()


def get_layer_names(model):
    layer_names = []
    for name, layer in model.named_children():
        layer_names.append((name, layer))
    return layer_names

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


def train (opt, kmeans_model, feature_extractor, train_loader):
    
    for i, data in enumerate(train_loader):
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),f'Epoch : {i}')
        with torch.no_grad():
            bz,c,h,w = data[0].shape
            batch_features = feature_extractor(data)
            kmeans_model.partial_fit(batch_features)
    feature_extractor.remove_hooks()
    
    return kmeans_model


def evaluate_kmeans(kmeans_model, val_loader):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Flatten the inputs if necessary
            batch_features = inputs.view(inputs.size(0), -1).numpy()
            
            # Predict the cluster for the current batch
            batch_predictions = kmeans_model.predict(batch_features)
            
            all_labels.extend(labels.numpy())
            all_predictions.extend(batch_predictions)
    
    return all_labels, all_predictions

if __name__ == '__main__':
    from options.train_options import TrainOptions
    from util import get_model
    import torch
    from sklearn.cluster import MiniBatchKMeans
    import time
    import pickle

    def get_train_opt():
        train_opt = TrainOptions().parse()
        train_opt.dataroot = 'datasets/ForenSynths_train_val'
        train_opt.detect_method = 'local_grad'
        train_opt.batch_size = 1024
        train_opt.num_threads = 1
        train_opt.kmean_model_name  ='kmeans_model'
        return train_opt
    
    def get_val_opt():
        val_opt = TrainOptions().parse(print_options=False)
        val_opt.dataroot = 'datasets/ForenSynths_train_val'
        val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
        val_opt.isTrain = False
        val_opt.no_resize = False
        val_opt.no_crop = False
        val_opt.serial_batches = True
        val_opt.classes = []
        val_opt.num_threads = 1
        return val_opt
    
    train_opt = get_train_opt()
    train_loader = create_dataloader(opt)
    model = resnet50_local_grad(pretrained=True, num_classes=1)
    feature_extractor =  FeatureExtractor(model, [model.avgpool])   
    
    #Create kmeans model
    kmeans_model = MiniBatchKMeans(n_clusters=5,
                                   random_state=0,
                                   batch_size=opt.batch_size,
                                   n_init="auto")

    
    #Save Kmeans model after trained
    kmeans = train(opt=train_opt, 
                   kmeans_model=kmeans_model, 
                   feature_extractor=feature_extractor, 
                   train_loader=train_loader)  
    with open(f'{opt.kmean_model_name}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    

    
    
    
    
    

    