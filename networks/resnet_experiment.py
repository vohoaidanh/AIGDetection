import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import numpy as np
from PIL import Image
import json

__all__ = ['ResNet', 'resnet50_experiment_01', 'Preprocess']


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

from functools import partial
class Preprocess():
    
    config = {
        "kernel_size": (5,5),
        "sigma": (2.0, 2.0),     
        "radius": (2.0, 2.0), #only for fft_filter   
        "filter": "no_filter",
        }
    mask = torch.ones(size=(3,3,3))
    
    @classmethod
    def __repr__(cls):
        s = f'Experiment config is \n {20*"*"} \n {json.dumps(cls.config, indent=4)}\n {20*"*"}'
        return s
    
    @classmethod    
    def create_mask(self,im_size, radius):
      
        h, w = im_size
        center_y, center_x = h // 2, w // 2
        
        if radius==0:
            return torch.zeros(im_size)

        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        mask = dist_from_center <= torch.tensor(radius)
        
        extended_mask = mask.unsqueeze(0).repeat(3, 1, 1)

        return extended_mask.float()
    

    @classmethod
    def low_pass_filter(self,x , kernel_size, sigma, **kwargs):
        #sigma=50% kernelsize
        return transforms.GaussianBlur(kernel_size, sigma)(x)
    
    @classmethod 
    def high_pass_filter(self,x , kernel_size, sigma, **kwargs):
        #sigma=50% kernelsize
        x =  x - transforms.GaussianBlur(kernel_size, sigma)(x)
        
        min_vals = x.view(1,3,-1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        max_vals = x.view(1,3,-1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        x_normalized = (x - min_vals) / (max_vals - min_vals + 1e-6)  # Add a small epsilon to avoid division by zero
    
        return x_normalized
    
    @classmethod 
    def fft_high_pass_filter(self, x, **kwargs):
        _, _, h, w = x.shape
        img_size = (h, w)
        radius = kwargs.get('radius', 30)
        if self.mask.size != img_size:
            mask = self.create_mask(im_size=img_size, radius = radius)
        else:
            mask = self.create_mask(im_size=img_size, radius = radius)
            
        
        mask = mask.unsqueeze(0)
            
        x_fft = torch.fft.fftn(x, dim=(-2, -1))  # Biến đổi Fourier 2D
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        x_fft_shifted_filter = x_fft_shifted * (1-mask)
        x_fft_shifted_filter = torch.fft.ifftshift(x_fft_shifted_filter, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(x_fft_shifted_filter, dim=(-2, -1))
        return x_filtered
    
    @classmethod 
    def fft_low_pass_filter(self, x, **kwargs):
        _, _, h, w = x.shape
        img_size = (h, w)
        radius = kwargs.get('radius', 30)
        if self.mask.size != img_size:
            mask = self.create_mask(im_size=img_size, radius = radius)
        else:
            mask = self.create_mask(im_size=img_size, radius = radius)
            
        
        mask = mask.unsqueeze(0)
            
        x_fft = torch.fft.fftn(x, dim=(-2, -1))  # Biến đổi Fourier 2D
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        x_fft_shifted_filter = x_fft_shifted * (mask)
        x_fft_shifted_filter = torch.fft.ifftshift(x_fft_shifted_filter, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(x_fft_shifted_filter, dim=(-2, -1))
        return x_filtered
    
    @classmethod 
    def no_filter(self,x , kernel_size, sigma):
        return x
    
    
    @classmethod 
    def filter(self):
        _filter =  getattr(self, self.config['filter'])
        return partial(_filter, 
                       kernel_size = self.config['kernel_size'], 
                       sigma = self.config['sigma'],
                       radius = self.config['radius'])
        
    
    def to_image(self,tensor):
        image_np = tensor.cpu().clone().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)  # Change tensor from CxHxW to HxWxC
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        
        
        min_val = image_np.min()
        max_val = image_np.max()
        normalized_array = (image_np - min_val) / (max_val - min_val)
        
        # Convert to the range 0-255 and convert to uint8
        uint8_array = (normalized_array * 255).astype(np.uint8)
        
        # Convert to a PIL Image
        image = Image.fromarray(uint8_array)

        return image
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        #low pass filter
        self.pre_process = Preprocess.filter()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512, num_classes)

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

    def forward(self, x):
        
        x = self.pre_process(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


def resnet50_experiment_01(pretrained=False, **kwargs):
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

if __name__ == '__main__':
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    #print(json.dumps(Preprocess.config, indent=4))
    print(Preprocess())
    #model = resnet50_experiment_01()
    #x = torch.rand((4,3,224,224))
    #model(x)
    x = torch.rand((4,3,224,224))
    img = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\1_fake\01bfd85a-84ab-415c-8ba3-fec489ae7944.jpg")
    im = transforms.ToTensor()(img)
#    pre = model(x)    
    process = Preprocess()
    #mask = process.create_mask((100,100), 30) 
    #plt.imshow(mask[2,:,:], cmap='gray')
    im_blur_highpass = process.fft_high_pass_filter(im.unsqueeze(0), radius=90)
    im_blur_highpass = process.to_image(torch.abs(im_blur_highpass).squeeze(0))
    plt.imshow(im_blur_highpass)
    
    im_blur_lowpass = process.fft_low_pass_filter(im.unsqueeze(0), radius=90)
    im_blur_lowpass = process.to_image(torch.abs(im_blur_lowpass).squeeze(0))
    plt.imshow(im_blur_lowpass)

    im_blur_lowpass = process.to_image(torch.abs(im_blur_lowpass).squeeze(0) + torch.abs(im_blur_highpass).squeeze(0) )

    
    plt.imshow(np.asanyarray(im_blur_highpass) + np.asanyarray(im_blur_lowpass))

    im_blur = process.low_pass_filter(im.unsqueeze(0),kernel_size=(5,5), sigma=(2,2))
    im_blur = process.to_image(im_blur.squeeze(0))
    
    im_blur_highpass = process.high_pass_filter(im.unsqueeze(0),kernel_size=(5,5), sigma=(2,2))
    im_blur_highpass = process.to_image(im_blur_highpass.squeeze(0))

    im_arr = np.asarray(im_blur_highpass)
    
    im_arr[:,:,0].max()
        
    im = torch.randn(3, 256, 256)  # Example tensor, replace with your tensor
    im = im.view(3, -1)
    # Compute minimum values across dimensions 2 and 3, keeping dimensions
    min_vals = im.min(dim=1)[0]
    im.shape
    
    im_blur.view(1,3,-1).min(dim=2)[0].squeeze(0)
    im_blur.view(1,3,-1).shape
    
    # Vẽ hình ảnh
    plt.figure(figsize=(10, 4))  # Thiết lập kích thước của figure
    
    # Vẽ hình ảnh thứ nhất
    plt.subplot(1, 3, 1)  # Subplot đầu tiên trên 1 hàng, 3 cột
    plt.imshow(img)
    plt.axis('off')  # Tắt trục
    plt.title('(a)')  # Tiêu đề của hình ảnh
    
    # Vẽ hình ảnh thứ hai
    plt.subplot(1, 3, 2)  # Subplot thứ hai trên 1 hàng, 3 cột
    plt.imshow(im_blur)
    plt.axis('off')  # Tắt trục
    plt.title('(b)')  # Tiêu đề của hình ảnh
    
    # Vẽ hình ảnh thứ ba
    plt.subplot(1, 3, 3)  # Subplot thứ ba trên 1 hàng, 3 cột
    plt.imshow(im_blur_highpass)
    plt.axis('off')  # Tắt trục
    plt.title('(c)')  # Tiêu đề của hình ảnh
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Điều chỉnh khoảng cách giữa các subplot
    
    plt.tight_layout()  # Cân chỉnh layout
    plt.show()
    
    
    