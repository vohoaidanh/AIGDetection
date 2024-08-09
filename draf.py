import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split


from networks.resnet_gradient import resnet50_gradient


model = resnet50_gradient(model_A_path = r'D:/K32/do_an_tot_nghiep/NPR-DeepfakeDetection/Gaussblur-4class-resnet-car-cat-chair-horse2024_06_20_08_12_39_model_eopch_7_best.pth', 
                          pretrainedA=False, 
                          pretrainedB=False)


intens = torch.rand(2,3,5,5)
out = model(intens)


