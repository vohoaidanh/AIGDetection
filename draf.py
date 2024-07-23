from PIL import Image
import numpy as np

import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from copy import deepcopy
def center_crop(image, crop_size = (256,256)):
    # Calculate the coordinates for a center crop
    width, height = image.size
    left = (width - crop_size[0]) / 2
    top = (height - crop_size[1]) / 2
    right = (width + crop_size[0]) / 2
    bottom = (height + crop_size[1]) / 2
    return image.crop((left, top, right, bottom))

def slip_image(image_pil):
    if isinstance(image_pil, str):
        image_pil = Image.open(image_pil).convert('RGB')
    image_np = np.asarray(image_pil)
    odd_rows_cols_image = image_np[1::2, 1::2]
    even_rows_cols_image = image_np[::2, ::2]
    return odd_rows_cols_image, even_rows_cols_image
    


NOISE = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

def interpolate(img, factor=0.5):
    old_image = deepcopy(img)
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
   

def gradient_filter_(input_tensor):
    input_tensor = TF.to_tensor(input_tensor).unsqueeze(0)
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the gradient filters for x and y directions
    kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(2, 3)  # Transpose kernel_x to get kernel_y

    # Expand the kernels to match the number of input channels
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    # Apply the filters
    diff_x = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels)
    diff_y = F.conv2d(input_tensor, kernel_y, padding=1, groups=channels)

    # Add a small value to avoid division by zero
    diff_x = diff_x + 1e-9
    diff_y = diff_y + 1e-9

    # Calculate the final output
    output = (torch.arctan(diff_y / diff_x) / (torch.pi / 2) + 1.0) / 2.0

    return output

import random
from tqdm import tqdm

def calculate_spectrum(folder_path, size=(512, 512), channel=1):
    spectra_odd, spectra_even = [],[]
    all_files = []
    for root, _, files in os.walk(folder_path):
        all_files.extend(files)
    print(all_files)
    random.shuffle(all_files)
    for file in tqdm(all_files):
      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
          image_path = os.path.join(root, file)
          image = Image.open(image_path).convert('RGB')
          image = center_crop(image)
          image_odd, image_even = slip_image(image)
          #image_odd, image_even = image, image
          #image_odd = TF.to_pil_image(image_odd) #TF.to_tensor(image_odd)
          #image_odd = image_odd.unsqueeze(0)
          
          spectrum = interpolate(image_odd)
          spectra_odd.append(spectrum)
          
          #image_even =  TF.to_pil_image(image_even)#TF.to_tensor(image_even)
          #image_even = image_even.unsqueeze(0)
          spectrum = interpolate(image_even)
          spectra_even.append(spectrum)
          
          
    spectra_odd = np.array(spectra_odd)
    mean_spectrum_odd = np.mean(spectra_odd, axis=0)
    
    spectra_even = np.array(spectra_even)
    mean_spectrum_even = np.mean(spectra_even, axis=0)

    # Apply logarithmic scaling to enhance visibility
    #epsilon = 1e-9  # Small value to avoid log(0)
    #log_mean_spectrum = np.log2(mean_spectrum + epsilon)

    # Normalize to the range [0, 255]
    #normalized_spectrum = (log_mean_spectrum - np.min(log_mean_spectrum)) / (np.max(log_mean_spectrum) - np.min(log_mean_spectrum)) * 255

    return mean_spectrum_odd, mean_spectrum_even



def calculate_fourier_spectrum(image):
    #image = Image.open(image_path).convert('RGB')
    #image = center_crop(image, size)
    image_array = np.asarray(image)
    image_channel = image_array[:, :, 0]

    f_transform = fft2(image_channel)
    f_transform_shifted = fftshift(f_transform)


    magnitude_spectrum = np.abs(f_transform_shifted)

    return magnitude_spectrum

mean_spectrum_odd, mean_spectrum_even = calculate_spectrum(r'D:\Downloads\dataset\progan_val_4_class\cat\1_fake')



plt.imshow(np.log(mean_spectrum_odd+1e-5),cmap='gray')
plt.imshow(np.log(mean_spectrum_even+1e-9))


mean_spectrum_odd = mean_spectrum_odd.squeeze(0).transpose(1,2,0)
plt.imshow(mean_spectrum_odd)


mean_spectrum_odd.shape


image = Image.open(r"D:\Downloads\dataset\progan_val_4_class\car\0_real\03631.png").convert('RGB')
width, height = image.size
left = 0
upper = 0
right = width//2 * 2
lower = height//2 * 2

# Crop the image
image = image.crop((left, upper, right, lower))
im = image
im = TF.to_tensor(im)

a = interpolate(im)
a = interpolate(im.unsqueeze(0))


plt.imshow(torch.abs(((im-a.squeeze(0)+1e-9).log())).permute(1,2,0))



F.linear(torch.tensor([1,2,3,4],dtype=float), torch.tensor([1,1,1,1],dtype=float))



A = np.array([
    [.5, .5, .25],
    [.25, 0., .25],
    [.25, .5, .5]
    ])

print(A)

result = A
for i in range(50):
    result = np.dot(result,A)

print(result)

np.dot(result, np.array([0.15,0.3,0.55]))




import torch
height  = 100
width = 100
r = 30
y, x = torch.meshgrid(torch.arange(-height//2, height//2), torch.arange(-width//2, width//2), indexing="ij")
radius = torch.sqrt(x.float()**2 + y.float()**2)
high_pass_mask = (radius <= r).float()

plt.imshow(high_pass_mask, cmap='gray')
    
mask = torch.ones(size = (height,width)) * 0
    
    
def create_circle_mask(size = (15,15), r = 3):
    mask = torch.zeros(size = size) 
    center_y, center_x = size[0]//2, size[1]//2
    for i in range(size[0]):
        for j in range(size[1]):
            if ((i-center_y)**2 + (j-center_x)**2) <= r**2:
                mask[i,j] = 1
    return mask

def create_mask(im_size = (256,256), mask_size = (100,100)):
    matrix = torch.ones(mask_size)
    original_height, original_width = matrix.size()
    target_height, target_width = im_size
    pad_top = (target_height - original_height) // 2
    pad_bottom = target_height - original_height - pad_top
    pad_left = (target_width - original_width) // 2
    pad_right = target_width - original_width - pad_left

    padded_matrix = torch.nn.functional.pad(matrix, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    return padded_matrix
    
    

plt.imshow(create_mask(im_size = (256,512),  mask_size = (100,150)), cmap='gray')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_gaussian_mask(shape, sigma):
    """
    Create a Gaussian mask in the frequency domain.

    Args:
    shape (tuple): Shape of the mask (height, width).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    np.ndarray: Gaussian mask of shape (height, width).
    """
    h, w = shape
    center_y, center_x = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    gaussian_mask = np.exp(-(dist_from_center ** 2) / (2 * sigma ** 2))
    return gaussian_mask

def apply_gaussian_filter_rgb(image, sigma):
    """
    Apply a Gaussian filter to an RGB image using Fourier transform.

    Args:
    image (np.ndarray): Input RGB image with shape (height, width, channels).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    np.ndarray: Filtered RGB image.
    """
    filtered_channels = []
    for channel in range(image.shape[-1]):
        # Extract channel
        image_channel = image[..., channel]

        # Compute the 2D FFT of the channel
        image_fft = np.fft.fft2(image_channel)
        image_fft_shifted = np.fft.fftshift(image_fft)

        # Create a Gaussian mask
        gaussian_mask = create_gaussian_mask(image_channel.shape, sigma)

        # Apply the Gaussian mask in the frequency domain
        image_fft_shifted_filtered = image_fft_shifted * gaussian_mask

        # Perform the inverse FFT
        image_fft_filtered = np.fft.ifftshift(image_fft_shifted_filtered)
        filtered_channel = np.fft.ifft2(image_fft_filtered).real

        # Append filtered channel to list
        filtered_channels.append(filtered_channel)

    # Stack filtered channels back into RGB image
    filtered_image = np.stack(filtered_channels, axis=-1)
    return filtered_image

# Load an example RGB image
image = Image.open(r"D:\K32\do_an_tot_nghiep\data\real_gen_dataset\train\1_fake\01bfd85a-84ab-415c-8ba3-fec489ae7944.jpg")
image_np = np.array(image)

# Apply Gaussian filter with specified sigma
sigma = 1
filtered_image_np = apply_gaussian_filter_rgb(image_np, sigma)

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_image_np)
plt.axis('off')

plt.show()










import re

# Dữ liệu từ bảng 1
text1 = """
=================================
(0 biggan      ) acc: 67.1; ap: 71.4
(1 cyclegan    ) acc: 61.7; ap: 52.5
(2 deepfake    ) acc: 50.4; ap: 60.3
(3 gaugan      ) acc: 56.3; ap: 66.5
(4 progan      ) acc: 61.5; ap: 62.0
(5 stargan     ) acc: 74.8; ap: 73.8
(6 stylegan    ) acc: 59.8; ap: 58.6
(7 stylegan2   ) acc: 54.6; ap: 52.0
(8 Mean      ) acc: 60.8; ap: 62.2
*************************
"""

# Dữ liệu từ bảng 2 (giả định)
text2 = """
=================================
           ForenSynths
=================================
2024_07_08_09_10_09
(0 biggan      ) acc: 86.9; ap: 92.5
(2 cyclegan    ) acc: 97.0; ap: 98.8
(3 deepfake    ) acc: 72.2; ap: 93.7
(4 gaugan      ) acc: 83.2; ap: 85.6
(6 progan      ) acc: 99.9; ap: 100.0
(9 stargan     ) acc: 100.0; ap: 100.0
(10 stylegan    ) acc: 96.7; ap: 100.0
(11 stylegan2   ) acc: 98.7; ap: 100.0
(13 Mean      ) acc: 78.8; ap: 83.4
*************************
"""

# Regular expression pattern to extract acc values
pattern = r'acc:\s+([\d.]+)'

# Extract acc values from text1
matches1 = re.findall(pattern, text1)
acc1 = [float(match) for match in matches1]

# Extract acc values from text2
matches2 = re.findall(pattern, text2)
acc2 = [float(match) for match in matches2]

# List of algorithms (assuming they are in the same order for both tables)
algorithms = [
    'biggan', 'cyclegan', 'deepfake', 'gaugan',
    'progan', 'stargan', 'stylegan', 'stylegan2', 'Mean'
]


import matplotlib.pyplot as plt
import numpy as np

# Độ rộng của các thanh
bar_width = 0.35

# Vị trí của các thanh
index = np.arange(len(algorithms))

# Vẽ biểu đồ so sánh
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(index - bar_width/2, acc1, bar_width, label='High pass')
bars2 = ax.bar(index + bar_width/2, acc2, bar_width, label='Low pass')

# Đặt nhãn trục x, nhãn của công tắc chuẩn
ax.set_xlabel('Algorithms')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Accuracy between Table 1 and Table 2')
ax.set_xticks(index)
ax.set_xticklabels(algorithms, rotation=45, ha='right')
ax.legend()

# Chỉnh sửa càng
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu phân phối chuẩn với trung bình 0 và độ lệch chuẩn 1
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # ma trận hiệp phương sai (covariance matrix)

# Tạo 1000 điểm dữ liệu
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# Vẽ biểu đồ phân bố các điểm
plt.scatter(x, y, alpha=0.3)
plt.title('Gaussian Distribution of 1000 points')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.grid(True)
plt.show()



F.sigmoid(torch.tensor(-2.0))

import torch.nn.functional as F

features1 = torch.randn(3, 512)
features2 = torch.randn(3, 512)

# Calculate cosine similarity
distance = F.cosine_similarity(features1, features2)
distance = distance.unsqueeze(1)




import matplotlib.pyplot as plt
import numpy as np

# Hàm biến đổi
def sin_transform(x, a, b):
  return np.sin(x) + a * np.sin(a*x + b)

# Tham số
x_min = 0
x_max = 10
num_points = 1000
a = 2  # Tham số điều chỉnh độ dốc của sin(ax + b)
b = 0  # Tham số điều chỉnh vị trí dọc trục y của sin(ax + b)

# Tạo dữ liệu
x = np.linspace(x_min, x_max, num_points)
y1 = np.sin(x)  # sin(x)
y2 = sin_transform(x, a, b)  # sin(ax + b)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Vẽ đường sin(x)
plt.plot(x, y1, label='sin(x)', color='blue')

# Vẽ đường sin(ax + b)
plt.plot(x, y2, label='sin(ax + b)', color='red')

# Thêm chú thích
plt.title('Phân phối dữ liệu sin(x) và sin(ax + b)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Hiển thị đồ thị
plt.grid(True)
plt.tight_layout()
plt.show()


import torch

# Ví dụ đơn giản
x = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])  # Batch có 2 điểm dữ liệu, mỗi điểm có 2 chiều
centers = torch.tensor([[1.0, 1.0], 
                        [2.0, 2.0], 
                        [3.0, 3.0]])  # Centers của 3 lớp, mỗi center có 2 chiều
num_classes = 3  # Số lớp

# Tính toán khoảng cách Euclidean bình phương
batch_size = x.size(0)
distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
          torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()

print(distmat)
distmat.addmm_(1, -2, x, centers.t())


classes = torch.arange(2).long()

labels = torch.tensor([0,1,0])
labels = labels.unsqueeze(1).expand(3, 2)

labels.eq(classes.expand(3, 2))



import json
import cv2
filename = r"D:\Downloads\Bird Nest With AI Images.v2-without-preprocessing.coco\train\_annotations.coco.json"
with open(filename, 'r') as file:
    data = json.load(file)

filename = r"D:\Downloads\Bird Nest With AI Images.v2-without-preprocessing.coco\train\BrokenBig-16-_bmp.rf.bb3938008259536c39e333859d66cb91.jpg"
img = Image.open(filename)
img = img.resize(size=(400,400))
grayscale_image = img.convert('L')  # 'L' mode is for grayscale
# Hiển thị ảnh kết hợp
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.title('Image Origin')
plt.axis('off')
plt.show()

threshold = 50  # You can adjust this value
binary_image = grayscale_image.point(lambda x: 255 if x > threshold else 0, mode='1')
mask = np.asarray(binary_image)
mask = 1-mask

#plt.imshow(grayscale_image, cmap='gray')
#plt.imshow(grayscale_image*mask, cmap='gray')

kernel = np.ones((5, 5), np.uint8)
binary_image = mask*1.0
dilated_image = cv2.dilate(binary_image, kernel, iterations=3)
eroded_image = cv2.erode(dilated_image, kernel, iterations=3)
#plt.imshow(dilated_image, cmap='gray')
#plt.imshow(eroded_image, cmap='gray')

mask_rgb = (eroded_image * 255).astype(np.uint8)
mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2RGB)

image_rgb = np.asarray(img)

# Áp dụng độ trong suốt alpha lên mặt nạ
alpha = 0.5  # Độ trong suốt, có thể điều chỉnh từ 0 (trong suốt) đến 1 (không trong suốt)
mask_colored = np.zeros_like(mask_rgb)
mask_colored[mask > 0] = [0, 255, 0]  # Đặt màu cho vùng mặt nạ (ở đây là màu xanh lá cây)

# Kết hợp ảnh gốc và mặt nạ với độ trong suốt alpha
combined = cv2.addWeighted(image_rgb, 1, mask_colored, alpha, 0)

# Hiển thị ảnh kết hợp
plt.figure(figsize=(10, 10))
plt.imshow(combined)
plt.title('Image with Mask Overlay')
plt.axis('off')
plt.show()














