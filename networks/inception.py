# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights
from networks.local_grad import gradient_filter
#import torch.nn.functional as F

def split_and_shuffle_image_tensor(image_tensor, grid_size=8):
    # Lấy kích thước của ảnh
    batch_size, channels, height, width = image_tensor.shape
    
    # Chia ảnh thành các ô
    cell_height = height // grid_size
    cell_width = width // grid_size
    
    # Chia tensor ảnh thành các ô nhỏ
    cells = image_tensor.unfold(2, cell_height, cell_height).unfold(3, cell_width, cell_width)
    
    # Chuyển các ô thành danh sách để dễ dàng xáo trộn
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, -1, channels, cell_height, cell_width)
    
    # Xáo trộn danh sách các ô
    shuffled_indices = torch.randperm(cells.size(1))
    shuffled_cells = cells[:, shuffled_indices]
    
    # Ghép các ô đã xáo trộn lại thành ảnh
    shuffled_image = shuffled_cells.view(batch_size, grid_size, grid_size, channels, cell_height, cell_width)
    shuffled_image = shuffled_image.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch_size, channels, height, width)
    
    return shuffled_image

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
        self.preprocess = split_and_shuffle_image_tensor
        self.gradient_filter = gradient_filter
        input_dim = 80
        self.features = nn.Sequential(*list(original_model.children())[:5])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = MLPHead(input_dim=input_dim, output_dim=num_classes)
        
    def forward(self,x):
        x = self.preprocess(x)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


