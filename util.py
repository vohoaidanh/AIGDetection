import sys
import os
import torch
from torch import nn
import torchvision
from networks.resnet import resnet50
from networks.resnet_local_grad import resnet50_local_grad
from networks.resnet_1layer import resnet50_1layer

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        
        
def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)
    
    
def get_model(opt):
    if opt.detect_method in ["NPR"]:
        print(f'Detect method model {opt.detect_method}')
        model = resnet50(pretrained=False, num_classes=1)
        return model
        
    elif opt.detect_method.lower() in ['local_grad']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet50_local_grad(pretrained=True, num_classes=1)
        return model
    
    elif opt.detect_method.lower() in ['resnet_1layer']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet50_1layer(pretrained=False, num_classes=1)
        return model
    
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")
        
        
        
        
        
        
        
        
        
        
        
        
        
        