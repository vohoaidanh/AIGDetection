import sys
import os
import torch
import torch.nn as nn
#from torch import nn
#import torchvision
from networks.resnet import resnet50
from networks.resnet_local_grad import resnet50_local_grad
from networks.resnet_gradient import resnet50_gradient

#from networks.resnet_experiment import *
from networks.resnet_fusion import resnet50_fusion
from networks.resnet_lstm import resnet50_lstm
from networks.resnet_attention import simple_vit, pretrain_vit, finetun_vit_lora
from networks.semi_supervisor import resnet_similarity, resnet_center_loss
from networks.resnet_kmeans import resnet50_multi_branch
from networks.inception import inception_local_grad
from networks.main_models import build_model
from networks.center_loss import CenterLoss
from networks.resnet_attention_embedding import resnet50_text_combine


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
        #model = inception_local_grad(pretrained=True, num_classes=1)
        return model
    
    elif opt.detect_method.lower() in ['experiment']:
        print(f'Detect method model {opt.detect_method}')
        #model = resnet50_experiment_01(pretrained=False, num_classes=1)
        #Preprocess is contains the experiment configuration
        #model = build_model(backbone_name='vgg16', num_classes=1, layer_to_extract=-1)
        model = resnet50_text_combine(num_classes=1)

        return model
    
    elif opt.detect_method.lower() in ['model_fusion']:
        print(f'Detect method model {opt.detect_method}')
        opt.model1_path = r'model_epoch_last_3090.pth' # for NPR checkpoint
        opt.model2_path = 'weights\Gaussblur-4class-resnet-car-cat-chair-horse2024_06_20_08_12_39_model_eopch_7_best.pth'                           #For local grad checkpoints
        model = resnet50_fusion(num_classes=1, model1_path=opt.model1_path, model2_path=opt.model2_path)
        return model
    
    elif opt.detect_method.lower() in ['resnet_lstm']:
        print(f'Detect method model {opt.detect_method}')
        opt.backbone_path = r'Gaussblur-4class-resnet-car-cat-chair-horse2024_06_20_08_12_39_model_eopch_7_best.pth'
        model = resnet50_lstm(backbone_path=opt.backbone_path)
        return model
    
    elif opt.detect_method.lower() in ['resnet_similarity']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet_similarity(pretrained=True)
        return model
    
    elif opt.detect_method.lower() in ['resnet_center_loss']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet_center_loss(pretrained=True)
        return model
    
    elif opt.detect_method.lower() in ['resnet_kmeans']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet50_multi_branch(pretrained=True)
        return model
    
    elif opt.detect_method.lower() in ['vit']:
        print(f'Detect method model {opt.detect_method}')
        #model = simple_vit(num_classes=1, embedding_dim=256, mlp_dim=256)
        #model = pretrain_vit()
        model = finetun_vit_lora()
        return model
    
    elif opt.detect_method.lower() in ['gradient']:
        print(f'Detect method model {opt.detect_method}')
        model = resnet50_gradient(model_A_path='/workspace/AIGDetection/Gaussblur-4class-resnet-car-cat-chair-horse2024_06_20_08_12_39_model_eopch_7_best.pth', pretrainedA=True, pretrainedB=True)
        #model = pretrain_vit()
        return model

    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")
        
        

        
        
        
    
        
        
        
        
        