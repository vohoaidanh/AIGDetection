import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2

class GradientLayer(nn.Module):
    def __init__(self):
        super(GradientLayer, self).__init__()

        # Define the gradient filters for x and y directions
        kernel_x = torch.tensor([[-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1], [1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Expand the kernels to match the number of input channels (3)
        self.kernel_x = kernel_x.expand(1, 3, 1, 2)
        self.kernel_y = kernel_y.expand(1, 3, 2, 1)
        self.kernel_x = self.kernel_x.cuda()
        self.kernel_y = self.kernel_y.cuda()
        # Register the kernels as buffers (not updated during optimization)
        self.register_buffer('buffer_kernel_x', self.kernel_x)
        self.register_buffer('buffer_kernel_y', self.kernel_y)

    def forward(self, x):
        x = x.cuda()
        # Apply the filters separately to each channel
        diff_x = F.conv2d(x, self.kernel_x, padding=(0, 1))
        diff_y = F.conv2d(x, self.kernel_y, padding=(1, 0))

        # Combine the gradients
        print(self.kernel_x.shape)
        print(self.kernel_y.shape)
        combined_grad = torch.sqrt(diff_x**2 + diff_y**2)

        return combined_grad

def gradient_filter(input_tensor):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the gradient filters for x and y directions
    kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(2, 3)  # Transpose kernel_x to get kernel_y

    # Expand the kernels to match the number of input channels
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    # Apply the filters
    diff_x = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels) + 1e-9
    diff_y = F.conv2d(input_tensor, kernel_y, padding=1, groups=channels)
    
    diff = diff_y/diff_x
    diff = torch.where(torch.abs(diff) > 1e2, torch.tensor(0.0), diff)

    # Add a small value to avoid division by zero
    #diff_x = diff_x + 1e-9
    #diff_y = diff_y + 1e-9

    # Calculate the final output and normalize to [0..1]
    output = (torch.arctan(diff) / (torch.pi / 2) + 1.0) / 2.0
    #output = torch.arctan2(diff_y, diff_x)
    #output = torch.arctan2(diff_y , (diff_x))
    #output2 = (torch.arctan2(diff_y , diff_x) / (torch.pi / 2) + 1.0) / 2.0
    return output

def gauss_blur(input_tensor):
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5))
    return blur(input_tensor)

def sobel(input_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
 
    sobel_x_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_x.to(input_tensor.device), padding=1)  # Sobel x
    sobel_y_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_y.to(input_tensor.device), padding=1)  # Sobel y

    mask = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    mask = mask.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
    mask = (mask>2.0*mask.mean()).float()
    mask = TF.gaussian_blur(mask,3,1.0)
    mask = (mask>0.0).float()
 
    out1 = TF.gaussian_blur(input_tensor,3,1.0)
    out1 = out1 * (mask)
    out2 = input_tensor*(1-mask)
    return out2 + out1

def edge_blur(input_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
 
    sobel_x_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_x.to(input_tensor.device), padding=1)  # Sobel x
    sobel_y_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_y.to(input_tensor.device), padding=1)  # Sobel y

    mask = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    mask = mask.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
    mask = (mask>2.0*mask.mean()).float()
    mask = TF.gaussian_blur(mask,3,1.0)
    mask = (mask>0.0).float()
 
    out1 = TF.gaussian_blur(input_tensor,3,1.0)
    out1 = out1 * (mask)
    out2 = input_tensor*(1-mask)
    return out2 + out1

def flat_blur(input_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
 
    sobel_x_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_x.to(input_tensor.device), padding=1)  # Sobel x
    sobel_y_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_y.to(input_tensor.device), padding=1)  # Sobel y

    mask = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    mask = mask.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
    mask = (mask>2.0*mask.median()).float()
    mask = TF.gaussian_blur(mask,3,1.0)
    mask = (mask>0.0).float()
 
    out1 = TF.gaussian_blur(input_tensor,3,1.0)
    out1 = out1 * (1-mask)
    out2 = input_tensor*(mask)
    return out2 + out1

def high_pass_mask(input_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
 
    sobel_x_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_x.to(input_tensor.device), padding=1)  # Sobel x
    sobel_y_output = F.conv2d(input_tensor[:,0:1,:,:], sobel_y.to(input_tensor.device), padding=1)  # Sobel y

    mask = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    mask = mask.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
    mask = (mask>2.0*mask.median()).float()
    #mask = TF.gaussian_blur(mask,3,1.0)
    mask = (mask>0.0).float()
 
    #out1 = TF.gaussian_blur(input_tensor,3,1.0)
    #out1 = out1 * (1-mask)
    #out2 = input_tensor*(mask)
    return mask#out2 + out1


if __name__ == '__main__':
    import torch
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    img = Image.open(r"D:\Downloads\dataset\progan_val_4_class\cat\0_real\17262.png")
    im = transforms.ToTensor()(img).unsqueeze(0)

    out_put = gradient_filter(im)
    out_put = out_put.squeeze(0)
    
    im_out = out_put.permute(1,2,0).numpy()
    im_out = (im_out * 255).astype(np.uint8)

    im_out = Image.fromarray(im_out)

    plt.imshow(out_put.permute(1,2,0))
    
    
    plt.figure(figsize=(7, 4))  # Thiết lập kích thước của figure
    
    # Vẽ hình ảnh thứ nhất
    plt.subplot(1, 2, 1)  # Subplot đầu tiên trên 1 hàng, 3 cột
    plt.imshow(img)
    plt.axis('off')  # Tắt trục
    plt.title('(a)')  # Tiêu đề của hình ảnh
    
    # Vẽ hình ảnh thứ hai
    plt.subplot(1, 2, 2)  # Subplot thứ hai trên 1 hàng, 3 cột
    plt.imshow(im_out)
    plt.axis('off')  # Tắt trục
    plt.title('(b)')  # Tiêu đề của hình ảnh

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Điều chỉnh khoảng cách giữa các subplot
    
    plt.tight_layout()  # Cân chỉnh layout
    plt.show()
    
    
    