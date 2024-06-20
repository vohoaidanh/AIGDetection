# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def apply_gaussian_blur(image, kernel):
    """Apply Gaussian blur to an image using a given kernel."""
    # Get dimensions of the image and kernel
    img_h, img_w = image.shape[:2]
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    
    # Pad the image to handle the edges
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    
    # Create an empty output image
    blurred_image = np.zeros_like(image)
    
    # Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            for k in range(image.shape[2]):  # For each channel
                blurred_image[i, j, k] = np.sum(
                    kernel * padded_image[i:i+kernel_size, j:j+kernel_size, k]
                )
    
    return blurred_image




# Tạo một ảnh mẫu (ví dụ: một ảnh đen trắng nhỏ)
#image = np.random.rand(256, 256)
image_path = r"D:\dataset\real_gen_dataset\train\0_real\000609337.jpg"
image = Image.open(image_path)
image = image.resize((300,300))
image = np.asarray(image)/255

k = create_gaussian_kernel(3,1)
k_fft = np.fft.fft2(k)

image_blur = apply_gaussian_blur(image,k)

# Thực hiện biến đổi Fourier nhanh 2D (FFT 2D)
fft2_result = np.fft.fft2(image)
fft2_shifted = np.fft.fftshift(fft2_result)  # Dịch chuyển zero-frequency component về trung tâm

# Trực quan hóa ảnh ban đầu và phổ của nó
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh ban đầu')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(fft2_shifted) + 1), cmap='gray')
plt.title('Phổ ảnh (biến đổi Fourier)')
plt.axis('off')

plt.tight_layout()
plt.show()


plt.imshow(image_blur, cmap='gray')



import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
# Example usage on a single image tensor
image_path = r"D:\dataset\real_gen_dataset\train\1_fake\01ae3103-c284-4367-b593-1185741e1766.jpg"
image_pil = Image.open(image_path).convert('RGB')  # Load image as PIL.Image

image = np.asarray(image_pil)
# Apply Laplacian filter
# Apply Sobel filters
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel x-direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel y-direction

# Convert the output to uint8 (absolute values and scaling)
sobel_x_abs = np.uint8(np.absolute(sobel_x))
sobel_y_abs = np.uint8(np.absolute(sobel_y))

# Combine Sobel x and Sobel y outputs to get edge magnitude
sobel_combined = cv2.bitwise_or(sobel_x_abs, sobel_y_abs)
plt.imshow(sobel_combined)

#plt.imshow(image*mask)

intensor = TF.to_tensor(image_pil)
intensor = intensor.unsqueeze(0)
#intensor = torch.rand((2,3,224,224))
size = 3
sigma = 0.5
GAUSS_KERNEL = torch.Tensor(np.outer(np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)),
                                   np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)))).view(1, 1, size, size)

def sobel(input_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
 
    sobel_x_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_x.to(input_tensor.device), padding=1)  # Sobel x
    sobel_y_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_y.to(input_tensor.device), padding=1)  # Sobel y

    mask = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    mask = mask.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
    mask = (mask>2.0*mask.mean()).float()
    mask = TF.gaussian_blur(mask,3,2.0)
    mask = (mask>0.0).float()
 

    #blurred1 = F.conv2d(input_tensor[:,0:1,:,:], GAUSS_KERNEL, padding=size//2)
    #blurred2 = F.conv2d(input_tensor[:,1:2,:,:], GAUSS_KERNEL, padding=size//2)
    #blurred3 = F.conv2d(input_tensor[:,2:3,:,:], GAUSS_KERNEL, padding=size//2)
    
    out1 = TF.gaussian_blur(input_tensor,15,1.0)
    out1 = out1 * (1-mask)
    #out1 = torch.cat((blurred1, blurred2, blurred3), dim=1)#*mask

    out2 = input_tensor*(mask)
    #blurred = torch.cat((blurred1, blurred2, blurred3), dim=1)

    return out2 + out1


out  = sobel(intensor)
out = out.view(-1,out.shape[-2],out.shape[-1])
plt.imshow(out.permute(1,2,0))

import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
import numpy as np
from networks.local_grad import *
import torch
from torchvision import transforms

window = [-0, 1, 0,
          1, 4, 1,
          -0, 1, 0]
kernel = ImageFilter.Kernel(size=(3,3), kernel = window,scale=8)
image_path = r"D:\Downloads\dataset\cat\1_fake\14092.png"
image_pil = Image.open(image_path).convert('RGB')  # Load image as PIL.Image
image_pil_resize = image_pil.resize((image_pil.height//2,image_pil.width//2)).resize(image_pil.size)
#im_blur = image_pil.filter(kernel)

diff = np.asarray(image_pil) - np.asarray(image_pil_resize)
plt.imshow(diff)


imtensor = transforms.ToTensor()(image_pil)
im_gauss = gradient_filter(imtensor.unsqueeze(0))
#diff = (imtensor - im_gauss.squeeze(0))
diff =im_gauss.squeeze(0)
diff = (diff-diff.min())/(diff.max()-diff.min())*255
diff = np.asarray(diff.permute(1,2,0), dtype='uint8')
plt.imshow(diff)

diff = Image.fromarray(diff)
diff = np.asarray(diff)
plt.imshow(im_gauss.squeeze(0).permute(1,2,0))
plt.imshow(diff)

image_pil_resize.height

plt.imshow(diff.permute(1,2,0))


torch.tan(torch.tensor(1/1e-9))



import torch
import util
from options.train_options import TrainOptions

# Create a tensor with values covering a range including ±π/2
input_tensor = torch.tensor([-torch.pi, -torch.pi/2, 0, torch.pi/2, torch.pi], dtype=torch.float32)

# Apply torch.tan
output_tensor = torch.tan(input_tensor)


torch.arctan(torch.tensor(-1.5708))

(diff/(torch.pi/2)).min()
imtensor.max()

opt = TrainOptions().parse()
opt.detect_method = 'NPR'
model = util.get_model(opt)

model_dict = torch.load('model_epoch_last_3090.pth', map_location='cpu')

model_dict = model_dict['model']
model_dict.keys()

model.load_state_dict(model_dict)



# Giả sử model_dict là state_dict của model của bạn đã được lưu
model_dict = torch.load('NPR.pth', map_location='cpu')

# Tạo một dictionary mới để chứa state_dict đã được chỉnh sửa
new_model_dict = {}
for key, value in model_dict.items():
    # Loại bỏ phần "module." ở đầu mỗi key
    new_key = key.replace("module.", "", 1)
    new_model_dict[new_key] = value




model.load_state_dict(new_model_dict['model'])






