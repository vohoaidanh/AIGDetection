import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

def combine_fft(main_img, ref_img):
    
    # Thực hiện biến đổi Fourier cho ảnh thứ nhất
    fft_img1 = torch.fft.fft2(main_img)
    magnitude_img1 = torch.abs(fft_img1)
    phase_img1 = torch.angle(fft_img1)
    
    # Thực hiện biến đổi Fourier cho ảnh thứ hai
    fft_img2 = torch.fft.fft2(ref_img)
    magnitude_img2 = torch.abs(fft_img2)
    phase_img2 = torch.angle(fft_img2)
    
    
    real = magnitude_img2 * torch.cos(phase_img2)
    imag = magnitude_img2 * torch.sin(phase_img2)
    
    combined_fft = torch.complex(real, imag)
    combined_image = torch.fft.ifft2(combined_fft)

    return torch.abs(combined_image)


def combine_magnitude_and_phase(magnitude_img, phase_img):
    # Đảm bảo các tensor có kích thước 2D
    if len(magnitude_img.shape) != 2 or len(phase_img.shape) != 2:
        raise ValueError("Cả magnitude và phase đều phải là tensor 2D")

    # Thực hiện biến đổi Fourier cho ảnh pha để lấy phổ Fourier
    #phase_img_fft = torch.fft.fft2(phase_img)

    # Kết hợp magnitude của ảnh thứ hai với pha của ảnh thứ nhất
    real = magnitude_img * torch.cos(phase_img)
    imag = magnitude_img * torch.sin(phase_img)

    # Tạo phổ Fourier mới
    combined_fft = torch.complex(real, imag)

    # Thực hiện biến đổi ngược Fourier để lấy ảnh kết hợp
    combined_image = torch.fft.ifft2(combined_fft)

    return torch.abs(combined_image)

def plot_image(tensor_image, title='Image'):
    # Chuyển đổi tensor thành numpy để sử dụng với matplotlib
    plt.imshow(tensor_image.numpy(), cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
def load_image_as_tensor(image_path):
    # Đọc ảnh bằng PIL
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224,224))
    # Định nghĩa transform để chuyển ảnh thành tensor
    transform = transforms.Compose([
        transforms.ToTensor()  # Chuyển đổi ảnh thành tensor
    ])
    
    # Áp dụng transform để chuyển đổi ảnh thành tensor
    image_tensor = transform(image)
    
    return image_tensor
# Tạo hai tensor ảnh mẫu
#img1 = torch.randn(256, 256)  # Ảnh thứ nhất
#img2 = torch.randn(256, 256)  # Ảnh thứ hai

# Đọc ảnh và chuyển đổi thành tensor
img1 = r"D:\Downloads\dataset\CNN_synth\deepfake\0_real\485_0702.png"
img1 = load_image_as_tensor(img1)
# Đọc ảnh và chuyển đổi thành tensor
img2 = r"D:\Downloads\dataset\CNN_synth\cyclegan\horse\1_fake\n02391049_2380_fake.png"
img2 = load_image_as_tensor(img2)


# Thực hiện biến đổi Fourier cho ảnh thứ nhất
fft_img1 = torch.fft.fft2(img1)
magnitude_img1 = torch.abs(fft_img1)
phase_img1 = torch.angle(fft_img1)

# Thực hiện biến đổi Fourier cho ảnh thứ hai
fft_img2 = torch.fft.fft2(img2)
magnitude_img2 = torch.abs(fft_img2)
phase_img2 = torch.angle(fft_img2)

# Kết hợp magnitude của ảnh thứ hai với pha của ảnh thứ nhất
combined_image = combine_magnitude_and_phase(magnitude_img1[0], phase_img2[0])

# Vẽ ảnh kết hợp
plot_image(combined_image, title='magnitude_img1 + phase2')

# Kết hợp magnitude của ảnh thứ hai với pha của ảnh thứ nhất
combined_image = combine_magnitude_and_phase(magnitude_img2[0], phase_img1[0])

# Vẽ ảnh kết hợp
plot_image(combined_image, title='magnitude_img2 + phase1')



combined_image = combine_magnitude_and_phase(magnitude_img1[0], phase_img1[0])
plot_image(combined_image, title='image1 real')

combined_image = combine_magnitude_and_phase(magnitude_img2[0], phase_img2[0])
plot_image(combined_image, title='image2 fake')

combined_image = combine_magnitude_and_phase( magnitude_img2[0] * (1-gaussian_matrix) , phase_img2[0])
plot_image((combined_image), title='fake one')

combined_image = combine_fft(img1, img2)
combined_image=combined_image.permute(1,2,0)
plot_image(combined_image, title='fake one')


s = samples[5]
s = s.permute(1,2,0)
plot_image(s, title='sample one')



import math

def gaussian_kernel(size: int, sigma: float):
    """Generate a 2D Gaussian kernel."""
    kernel = torch.zeros((size, size), dtype=torch.float32)
    center = size // 2

    for i in range(size):
        for j in range(size):
            dist = (i - center)**2 + (j - center)**2
            kernel[i, j] = math.exp(-dist / (2 * sigma**2))
    
    kernel /= kernel.sum()  # Normalize the kernel
    return kernel

# Example usage
size = 224
sigma = 56.0
gaussian_matrix = gaussian_kernel(size, sigma)
plt.imshow(gaussian_matrix)



import clip
import torch

# Tải mô hình CLIP
model, preprocess = clip.load("ViT-B/32", device="cpu")


# Duyệt qua các module và in ra những module có tên chứa "ln_2"
for name, module in model.visual.named_modules():
    if "ln_2" in name:
        print(name, module)
        print(20*'-')
    else:
        print(name, module)



from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(768,1,bias=True)
image = torch.rand(12,3,224,224)
inputs = processor(images=image, return_tensors="pt",do_rescale=False)

outputs = model(input_tensor)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])

input_tensor = inputs['pixel_values']


torch.max(input_tensor/12, dim=0)
import torch
import torch.nn as nn
import numpy as np
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


norm = nn.LayerNorm([4,4])
bnorm = nn.BatchNorm2d([4])

x = torch.rand(2,3,4,4)


y = norm(x).detach()
y[0][1].var()

import torchvision.models as models

class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model, extract_layer):
        super(ViTFeatureExtractor, self).__init__()
        self.vit_model = vit_model
        self.extract_layer = extract_layer

    def forward(self, x):
        # Forward pass through the model
        for name, layer in self.vit_model.named_children():
            print(name, 100*'-')
            x = layer(x)
            if name == self.extract_layer:
                break
        return x
    
vit_model = models.vit_b_16(weights='IMAGENET1K_V1')
#vit_model = models.vit_l_16(weights=None)

extract_layer = 'encoder_layer_11'  # Replace with the actual layer name
feature_extractor = ViTFeatureExtractor(vit_model, extract_layer)

intens = torch.rand(3,3,224,224)
with torch.no_grad():
    features = feature_extractor(intens)



for name, layer in vit_model.named_children():
    print(name, 20*'-')


class Hook:
    def __init__(self, model, layers_to_hook):
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.features = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers_to_hook:
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def clear_features(self):
        self.features = {}

    def __call__(self, x):
        self.clear_features()
        with torch.no_grad():
            _ = self.model(x)
        return self.features



N = 3  # Change this to the desired layer number
layers_to_hook = [f'encoder.layers.encoder_layer_{N}' for N in range(23)]  # Replace with actual layer path in the model

# Create the Hook instance
hook = Hook(vit_model, layers_to_hook)

dummy_input = torch.randn(1, 3, 224, 224)
features = hook(dummy_input)
feature_key = list(features.keys())[0]


features['encoder.layers.encoder_layer_0'] - features['encoder.layers.encoder_layer_1'] 

hook.remove_hooks()

32*32
































