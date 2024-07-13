# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
#import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
#from typing import Any, cast, Dict, List, Optional, Union
#import numpy as np
from networks.resnet_local_grad import resnet50_local_grad 
import json
#from functools import partial

import torchvision.transforms as transforms
#from PIL import Image

__all__ = ['ResNetLSTM', 'resnet50_lstm']


class Preprocess():
    
    @classmethod
    def __repr__(cls):
        #s = f'Experiment config is \n {20*"*"} \n {json.dumps(cls.config, indent=4)}\n {20*"*"}'
        s = 'this is Preprocess class'
        return s
    
    @classmethod 
    def blur(self, img_tensor, kernel_size):
        transform = transforms.GaussianBlur(kernel_size)
        return transform(img_tensor)
    
    @classmethod    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    @classmethod 
    def add_gaussian_noise(self, img, amplitude):
        device = img.device
        noise = torch.randn(img.size()) * amplitude
        noise = noise.to(device)
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        return noisy_img
    
    @classmethod 
    def process(self,x):
        x1 = self.blur(img_tensor=x, kernel_size=3)
        x2 = self.blur(img_tensor=x, kernel_size=5)
        x3 = self.blur(img_tensor=x, kernel_size=7)
        x4 = self.blur(img_tensor=x, kernel_size=9)
        combined_tensor = torch.stack([x,x1,x2,x3,x4], dim=1)  # Shape: [batch_size, sequence_length, channels, width, height]

        return combined_tensor
    
    

# Định nghĩa mô hình kết hợp ResNet và LSTM
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(ResNetLSTM, self).__init__()
        self.preprocess = Preprocess.process
        # ResNet để trích xuất đặc trưng từ ảnh
        self.resnet = resnet50_local_grad(pretrained=False)  
        self.resnet.fc1 = nn.Identity()  # Loại bỏ fully connected layer cuối cùng của ResNet
        
        # LSTM để học mối quan hệ giữa các đặc trưng
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        # Fully connected layer để phân loại
        self.fc = nn.Linear(hidden_size, num_classes)
        
                # Freeze parameters of models A and B
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        
    def get_trainable_params(self):
        # Get parameters that require gradients (those of the MLP)
        return filter(lambda p: p.requires_grad, self.parameters())
    
        
    def load_backbone_weights(self, backbone_path):
        # Load weights for model A
        try:            
            state_dict = torch.load(backbone_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
                
            state_dict = {k: v for k, v in state_dict.items() if 'fc1' not in k}  # Remove keys related to FC layer
            self.resnet.load_state_dict(state_dict, strict=False)
            print(f'load backbone weight: {backbone_path} successfully!')
        except Exception as e:
            print(f"Failed to load model : {e}")
        
    def forward(self, x):
        # Đầu vào x là batch gồm 5 hình ảnh (shape: batch_size x 5 x 3 x 224 x 224)
        #
        
        x = self.preprocess(x)
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)  # Reshape để truyền vào ResNet
        
        # Trích xuất đặc trưng từ ResNet
        features = self.resnet(x)  # shape: (batch_size * seq_length, 512)
        
        # Đưa về định dạng chuỗi để truyền vào LSTM
        features = features.view(batch_size, seq_length, -1)  # shape: (batch_size, seq_length, 512)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        
        # Lấy output của LSTM sau lớp cuối cùng
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        # Fully connected layer để phân loại
        output = self.fc(lstm_out)  # shape: (batch_size, num_classes)
        
        return output
    
def resnet50_lstm(backbone_path = None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1)
    hidden_size = kwargs.get('hidden_size', 256)
    num_layers = kwargs.get('num_layers', 1)
    model = ResNetLSTM(num_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers)
    
    if backbone_path is not None:
        model.load_backbone_weights(backbone_path)
    else:
        print('load model without backbone weight!!!')

    return model


# =============================================================================
# 
# # Khởi tạo mô hình
# num_classes = 1  # Số lớp phân loại
# hidden_size = 256  # Kích thước hidden state của LSTM
# num_layers = 1  # Số lớp LSTM
# 
# model = ResNetLSTM(num_classes, hidden_size, num_layers)
# 
# # Định nghĩa loss function và optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 
# # Ví dụ về cách sử dụng mô hình
# # Xác định đầu vào là batch gồm 5 hình ảnh cùng một class
# # Ví dụ: batch_size = 1, seq_length = 5, channels = 3 (RGB), height = 224, width = 224
# batch_input = torch.randn(2, 3, 224, 224)  # Đầu vào ngẫu nhiên, bạn cần thay đổi thành dữ liệu thực tế
# 
# # Feedforward
# outputs = model(batch_input)
# 
# # Tính toán loss và backpropagation
# targets = torch.tensor([0])  # Ví dụ: các nhãn cho batch_input
# loss = criterion(outputs, targets)
# loss.backward()
# optimizer.step()
# 
# print(outputs.shape)  # In ra shape của đầu ra để kiểm tra
# 
# im = Image.open(r"D:\dataset\biggan\0_real\817--n04285008_3950.png")
# intensor = F.to_tensor(im).unsqueeze(0)
# 
# process = Preprocess.process
# out = process(intensor)
# 
# out[0,0]
# out[0,4]
# 
# 
# 
# =============================================================================




