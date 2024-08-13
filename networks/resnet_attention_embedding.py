import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *
from einops import rearrange

__all__ = ['ResNet', 'resnet50_local_grad']

import torchvision.models as models

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


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5               
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, q=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        _, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
                
        if q is not None:
            q = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        else:
            q = k

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
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
        
        #self.att1 = Attention(dim=256, heads=8, dim_head=64)
        #self.pool1 = nn.AdaptiveAvgPool2d((1, 256))
    
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.att2 = Attention(dim=512, heads=8, dim_head=64)
        #self.pool2 = nn.AdaptiveAvgPool2d((1, 512))
        
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
    
    def freeze_layers(self):
    # Đóng băng các lớp cụ thể
        for layer in [self.conv1, self.bn1, self.maxpool, self.layer1]:
            for param in layer.parameters():
                param.requires_grad = False
        
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    
    def forward(self, x):
        x = self.gradient_layer(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
     
        #b, c, h, w = x.shape
        #x = x.view(b, c, -1).permute(0, 2, 1)
           
        #x = self.att1(x)
        #x = rearrange(x, 'b (h w) d -> b d h w', h = h, w=w)

        x = self.layer2(x)
       
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.att2(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h = h, w=w)
        
        x = self.avgpool(x)
        x = x.view(b, -1)
        x = self.fc1(x)
        
        return x

class SimpleAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.norm1000 = nn.LayerNorm(1000)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, q=None):
        
        batch_size, dim = x.shape
        seq_len = 1  # Nếu x là tensor 2D, seq_len sẽ là 1
  
        # Thêm chiều seq_len nếu cần thiết
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, dim] -> [batch_size, 1, dim]
     
        if q is not None and q.dim() == 2:
            q = q.unsqueeze(1)  # [batch_size, dim] -> [batch_size, 1, dim]
            
        # Chuẩn hóa x và q (nếu có)
        x = self.norm(x)
        if q is not None:
            #q = self.norm(q)
            q = self.norm1000(q)
        else:
            q = x

        # Tính toán Q từ q, và K, V từ x
        
        #Q = rearrange(self.to_q(q), 'b n (h d) -> b h n d', h=self.heads)
        Q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        KV = self.to_kv(x).chunk(2, dim=-1)
        K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), KV)

        # Tính toán attention scores
        dots = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # Áp dụng hàm softmax để tính attention weights
        attn = self.attend(dots)

        # Tính toán output bằng cách nhân attention weights với V
        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Áp dụng linear layer cuối cùng
        return self.to_out(out)
    
class Head(nn.Module):
    def __init__(self, num_classes=1):
        super(Head, self).__init__()
        self.att = SimpleAttention(dim=2048,heads=8, dim_head=125)
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)
        

    def forward(self, x, q):
        
        
        #b, c, h, w = x.shape
        #x = x.view(b, c, -1).permute(0, 2, 1)
        #q = q.view(b, c, -1).permute(0, 2, 1)
        
        out = self.att(x, q)
        
        #out = rearrange(out, 'b (h w) d -> b d h w', h = h, w=w)
        
        out = self.classifier(out)
                
        return out
        
    
    
    
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        #self.backbone.fc = nn.Identity()
        self.gradient_layer = gradient_filter  # Instantiate GradientLayer
        # Đóng băng các trọng số của backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()
        self.head = Head(1)
        

    def forward(self,x):
        self.backbone.eval()
        q = self.backbone(x)
        q = torch.softmax(q, dim=1)

        v = self.resnet(self.gradient_layer(x))

        out = self.head(v,q)
        out = out.view(-1,1)
        return out
    
    

import functools

def print_function_name(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
    

@print_function_name
def resnet50_text_combine(pretrained=False, **kwargs):
    model = MyModel(num_classes=1)
    
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
        
    #model.freeze_layers()
    #remind remove residual in Bottleneck
    return model




if __name__ == '__main__':
    from options.train_options import TrainOptions
    from util import get_model
    import torch
    #opt = TrainOptions().parse()
    #opt.detect_method = 'local_grad'
    #model = get_model(opt)
    model = resnet50_text_combine(pretrained=False)

    intens = torch.rand(2,3,224,224)
    out = model(intens)    

# =============================================================================
#     model.freeze_layers()
#     
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")    
#     
#     
# =============================================================================
    
    
    