import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.local_grad import *
from networks.resnet_local_grad import ResNet as ResNet
from networks.resnet_local_grad import Bottleneck as Bottleneck
from networks.inception import inception_local_grad

import sys
from data import create_dataloader
import pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['resnet50_multi_branch']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class MyResnetFeatures(ResNet):
    
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(MyResnetFeatures, self).__init__(block, layers, num_classes, zero_init_residual)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    
    def forward(self, x):
        x = self.gradient_layer(x)
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        feature_layer2 = self.layer2(x)
        feature_layer3 = self.layer3(feature_layer2)
        
        return feature_layer2, feature_layer3
    
class MyResnetCluster(ResNet):
    
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(MyResnetCluster, self).__init__(block, layers, num_classes, zero_init_residual)
    
    def forward(self, x):
        NPR  = x - self.interpolate(x, 0.5)
        x = self.conv1(NPR*2.0/3.0)
                
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
    

class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim, resnet_layer3):
        super(MLPHead, self).__init__()
        hidden_dim = 128
        self.layer3 = resnet_layer3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ResnetMultiBranch(nn.Module):
    def __init__(self, num_classes=1, kmeans_center=None, **kwargs):
        super(ResnetMultiBranch,self).__init__()
        self.kmeans_center = kmeans_center
        self.kmeans_center = self.kmeans_center.to(DEVICE)
        
        self.resnet = MyResnetFeatures(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.resnet_cluster = MyResnetCluster(Bottleneck, [3, 4, 6, 3], **kwargs)
        #self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.avgpool = self.resnet.avgpool
        self.avgpool_cluster = self.resnet_cluster.avgpool
        num_features = 1024#self.resnet.fc.in_features

        self.heads =nn.ModuleList([MLPHead(input_dim = num_features, output_dim = num_classes, resnet_layer3=self.resnet.layer3) for _ in range(len(self.kmeans_center))])

        self.freeze_layers()

    def freeze_layers(self):
        for name, param in self.resnet.named_parameters():
            #if not any(layer in name for layer in ['layer1', 'layer2', 'layer3']):
            param.requires_grad = False        
                
        for name, param in self.resnet_cluster.named_parameters():
            param.requires_grad = False

        # Ensure the heads are not frozen
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = True


    def forward(self, x):
        feature_layer3,_ = self.resnet(x)
        feature = self.resnet_cluster(x)
        feature = self.avgpool_cluster(feature)
        feature = feature.view(feature.size(0), -1)
        
        #feature_layer3 = self.avgpool(feature_layer3)
        #feature_layer3 = feature_layer3.view(feature_layer3.size(0), -1)
        
        distances = torch.norm(feature[:, None, :] - self.kmeans_center, dim=2)
        #print(f'distances=====================: {distances}')

        sum_distances  = torch.sum(distances, dim=1, keepdim=True)
        #print(f'max_distances=====================: {sum_distances}')

        w = distances/sum_distances
        w = w.t()
        _, max_indices = torch.max(w, dim=1)
        z = w.clone()  # Clone để không thay đổi tensor gốc
        
        z[torch.arange(w.size(0)), max_indices] *= 2.0

        #print(f'w=====================: {w.shape}\n sum_distances: {sum_distances}, {w}')

        
        outputs = torch.stack([head(feature_layer3) for head in self.heads], dim=0)  # Kích thước: (num_heads, batch_size, output_dim)
        outputs = outputs.view(len(self.kmeans_center), -1)
        #print(f'outputs=====================: {outputs.shape}, {outputs}')

        #w = w.unsqueeze(0).unsqueeze(-1)  # Kích thước: (1, batch_size, input_dim, 1)

        combined_output = (outputs * z).sum(dim=0)  # Kích thước: (batch_size, output_dim)

        return combined_output.view(-1,1)#, w.t()


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layers = layers
        self.feature_outputs = []

        # Register hooks for the specified layers
        self.hook_handles = []
        for layer in layers:
            handle = layer.register_forward_hook(self.hook_fn)
            self.hook_handles.append(handle)

    def hook_fn(self, module, input, output):
        self.feature_outputs.append(output)

    def forward(self, x):
        # Clear previous feature outputs
        self.feature_outputs = []
        
        # Perform forward pass through the model
        _ = self.model(x)
        
        return self.feature_outputs

    def remove_hooks(self):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
            
        


def get_layer_names(model):
    layer_names = []
    for name, layer in model.named_children():
        layer_names.append((name, layer))
    return layer_names

def resnet50_local_grad(pretrained=False, **kwargs):
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

def resnet50_multi_branch(pretrained=True, kmeans_path=None, **kwargs):
    
    #Load center
    if kmeans_path is None:
        kmeans_path = r'/workspace/AIGDetection/NPR_kmeans_model_progan_resnet_pretrain.pkl'
    with open(kmeans_path, 'rb') as file:
        print(f'resnet50_multi_branch Loading Kmeans centers at {kmeans_path}')
        kmeans = pickle.load(file)
        #kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
   
    
    model = ResnetMultiBranch(num_classes=1, kmeans_center=torch.tensor(kmeans.cluster_centers_))
    
    if pretrained:
        print(r'Loading pretrained resnet50 weight from model_zoo')
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict = {k: v for k, v in state_dict.items() if not any(layer in k for layer in ['fc', 'layer3', 'layer4'])}
        model.resnet.load_state_dict(new_state_dict,strict=False)
        model.resnet_cluster.load_state_dict(new_state_dict,strict=False)
    return model


def train (opt, kmeans_model, feature_extractor, train_loader):
    from tqdm import tqdm

    for data in tqdm(train_loader):
        #print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),f'Epoch : {i}')
        with torch.no_grad():
            in_tens = data[0].to(DEVICE)
            bz,c,h,w = data[0].shape
            batch_features = feature_extractor(in_tens)
            batch_features = batch_features[-1].view(bz,-1)
            batch_features_cpu = batch_features.cpu().numpy()
            kmeans_model.partial_fit(batch_features_cpu)
    feature_extractor.remove_hooks()
    
    return kmeans_model


def evaluate_kmeans(opt, kmeans_model, feature_extractor, val_loader):
    all_labels = []
    all_predictions = []
    #kmeans_model.cluster_centers_ = kmeans_model.cluster_centers_.astype(float)
   
    for i, data in enumerate(val_loader):
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),f'Epoch : {i}')
        with torch.no_grad():
            in_tens = data[0].to(DEVICE)
            bz,c,h,w = data[0].shape
            batch_features = feature_extractor(in_tens)
            batch_features = batch_features[-1].view(bz,-1)
            batch_features_cpu = batch_features.cpu().numpy()
            
            print(f'batch_features_cpu shape: {batch_features_cpu.shape}')
            #batch_features_cpu = torch.randn(10, 512, device=DEVICE)
            #batch_features_cpu = batch_features_cpu.cpu().numpy()
            
            y_pred = kmeans_model.predict(batch_features_cpu)
            labels = data[-1].numpy()
            all_labels.extend(labels)
            all_predictions.extend(y_pred)

            
    return all_labels, all_predictions




if __name__ == '__main__':
    
    print(50*'=')

    from options.train_options import TrainOptions
    from util import get_model
    import torch
    from sklearn.cluster import MiniBatchKMeans
    import time
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def get_train_opt():
        train_opt = TrainOptions().parse()
        train_opt.dataroot = 'D:\Downloads\dataset\progan_val_4_class'
        train_opt.detect_method = 'local_grad'
        train_opt.batch_size = 320
        train_opt.num_threads = 1
        train_opt.kmean_model_name  ='inception_kmeans'
        return train_opt
    
    
    train_opt = get_train_opt()
    train_loader = create_dataloader(train_opt)
    #model = resnet50_local_grad(pretrained=True, num_classes=1)
    model = inception_local_grad(pretrained=True, num_classes=1)
    
    #model.load_state_dict(torch.load(r'model_epoch_last_3090.pth', map_location='cpu'), strict=True)

    feature_extractor =  FeatureExtractor(model, [model.avgpool])   
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    #Create kmeans model
    kmeans_model = MiniBatchKMeans(n_clusters=5,
                                   random_state=0,
                                   batch_size=train_opt.batch_size,
                                   n_init="auto")

    
    #Save Kmeans model after trained
    kmeans = train(opt=train_opt, 
                   kmeans_model=kmeans_model, 
                   feature_extractor=feature_extractor, 
                   train_loader=train_loader)  
    with open(f'{train_opt.kmean_model_name}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    



    #Vadilation
    model = inception_local_grad(pretrained=True, num_classes=1)
    
    #model.load_state_dict(torch.load(r'model_epoch_last_3090.pth', map_location='cpu'), strict=True)

    feature_extractor =  FeatureExtractor(model, [model.avgpool])   
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    
    model_path = r"D:/K32/do_an_tot_nghiep/NPR-DeepfakeDetection/networks/inception_kmeans.pkl"
    with open(model_path, 'rb') as file:
        kmeans = pickle.load(file)
        #kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
        
    
    def get_val_opt():
        val_opt = TrainOptions().parse(print_options=True)
        val_opt.dataroot = r'D:\Downloads\dataset\CNN_synth'
        val_opt.val_split = r'cyclegan'
        val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
        val_opt.detect_method = 'local_grad'
        val_opt.batch_size = 64
        val_opt.isTrain = False
        val_opt.no_resize = False
        val_opt.no_crop = False
        val_opt.serial_batches = True
        val_opt.classes = []
        val_opt.num_threads = 1
        
        return val_opt

    val_opt = get_val_opt()
    val_opt.num_threads = 0
    val_loader = create_dataloader(val_opt)   
    
    all_labels, all_pred = evaluate_kmeans(opt=val_opt, 
                                           kmeans_model=kmeans, 
                                           feature_extractor=feature_extractor, 
                                           val_loader=val_loader)    

    df = pd.DataFrame(zip(all_labels, all_pred), columns=['Label', 'Value'])
    plt.figure(figsize=(10, 6))
    plt.title(val_opt.val_split)
    sns.countplot(data=df, x='Value', hue='Label')
    
    
# =============================================================================
# df = pd.DataFrame(zip(all_labels, all_pred), columns=['Label', 'Value'])
# 
# plt.figure(figsize=(10, 6))
# plt.title(val_opt.val_split)
# sns.countplot(data=df, x='Value', hue='Label')
# 
# 
# dataiter = iter(val_loader)  
# x = next(dataiter) 
# x0 = feature_extractor(x[0])
# x0 = x0[-1].view(64,-1)
# x0 = x0.cpu().detach().numpy()
# 
# cluster_centers = kmeans.cluster_centers_
# 
# distances = np.linalg.norm(x0[:, np.newaxis, :] - cluster_centers, axis=2)
# distances_norm = distances/(np.max(distances, axis=1).reshape(-1,1))
# distances_norm = 1.-distances_norm
# np.argmin(distances, axis=1)
# 
# kmeans.predict(x0)
# 
# kmeans_center = torch.tensor(cluster_centers)[:5]
# muti_head = ResnetMultiBranch(num_classes=1, kmeans_center=kmeans_center)
# 
# intens = torch.rand(6,3,224,224)
# out,w = muti_head(x[0])
# 
# w.t()[33]
# muti_head.kmeans_center
# 
# 
# 
# distances = torch.rand(3,5) 
# sum_distances = torch.sum(distances,dim=1)
# 
#     
# from networks.resnet import resnet50
# 
# mymodel = resnet50(pretrained=False, num_classes=1)
# out1 = mymodel(x[0])
# 
# out1[0]
# 
# 
# =============================================================================


# =============================================================================
# 
# kmeans_center = torch.rand(5,512)
# muti_head = ResnetMultiBranch(num_classes=1, kmeans_center=kmeans_center)
# 
# intens = torch.rand(4,3,224,224)
# out = muti_head(intens)
# =============================================================================
#out1 = model(intens)
# =============================================================================
# import pickle
# model = resnet50_multi_branch(pretrained=True, kmeans_center= torch.rand(5,512), 
#                               kmeans_path=r'D:/K32/do_an_tot_nghiep/NPR-DeepfakeDetection/kmeans_model_progan_resnet_pretrain.pkl')
# 
# 
# =============================================================================