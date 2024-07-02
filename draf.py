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









