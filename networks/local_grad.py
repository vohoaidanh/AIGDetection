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
    size = (input_tensor.shape[-2], input_tensor.shape[-1])
    # Define the gradient filters for x and y directions
    device = input_tensor.device
    kernel_x = torch.tensor([[0, 0, 0],[0, -1, 1],[0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Kernel for x direction
    kernel_y = kernel_x[0,0,:,:].t().unsqueeze(0).unsqueeze(0)  
    kernel_x = kernel_x.to(device)
    kernel_y = kernel_y.to(device)
    # Expand the kernels to match the number of input channels (3)
    #kernel_x = kernel_x.expand(-1, input_tensor.size(1), -1, -1)
    #kernel_y = kernel_y.expand(-1, input_tensor.size(1), -1, -1)

    # Apply the filters
    i=0
    diff_x1 = F.conv2d(input_tensor[:,i:i+1], kernel_x, padding=(1, 1), stride=1)
    diff_y1 = F.conv2d(input_tensor[:,i:i+1], kernel_y, padding=(1, 1))
    i=1
    diff_x2 = F.conv2d(input_tensor[:,i:i+1], kernel_x, padding=(1, 1))
    diff_y2 = F.conv2d(input_tensor[:,i:i+1], kernel_y, padding=(1, 1))

    i=2
    diff_x3 = F.conv2d(input_tensor[:,i:i+1], kernel_x, padding=(1, 1))
    diff_y3 = F.conv2d(input_tensor[:,i:i+1], kernel_y, padding=(1, 1))

    # Combine the gradients into a single tensor with 3 channels
    #combined_grad = torch.cat((diff_x, diff_y), dim=1)
    outx = torch.cat((diff_x1+1e-9, diff_x2+1e-9, diff_x3+1e-9), dim=1)
    outy = torch.cat((diff_y1+1e-9, diff_y2+1e-9, diff_y3+1e-9), dim=1)
    #outy = F.interpolate(outy, size)
    #out = torch.sqrt(outx**2 + outy**2)
    return (torch.arctan(outy/outx)/(torch.pi/2) + 1.0)/2.0     #torch.cat((diff_x1/diff_y1, diff_x2/diff_y2, diff_x3/diff_y1), dim=1)


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







