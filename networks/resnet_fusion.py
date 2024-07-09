# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.resnet import resnet50 as NPRModel
from networks.resnet_local_grad import resnet50_local_grad as LocalgradModel


class ModelFusion(nn.Module):
    def __init__(self, num_classes):
        super(ModelFusion, self).__init__()
        
        # Load pre-trained ResNet-50 models
        self.model_A =NPRModel(pretrained=False)
        self.model_B = LocalgradModel(pretrained=False)
        
        # Remove FC layer and add new FC layer
        in_features = self.model_A.fc1.in_features
        self.model_A.fc1 = nn.Identity()  # Remove FC layer of model A
        self.model_B.fc1 = nn.Identity()  # Remove FC layer of model B
        
        # MLP for fusion and classification
        self.head = nn.Sequential(
            nn.Linear(2 * in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
                
                # Freeze parameters of models A and B
        for param in self.model_A.parameters():
            param.requires_grad = False
        for param in self.model_B.parameters():
            param.requires_grad = False
            
    def get_trainable_params(self):
        # Get parameters that require gradients (those of the MLP)
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def load_weights(self, path_model_A, path_model_B):
        # Load weights for model A
        try:            
            state_dict_A = torch.load(path_model_A, map_location='cpu')
            if 'model' in state_dict_A:
                state_dict_A = state_dict_A['model']
                
            state_dict_A = {k: v for k, v in state_dict_A.items() if 'fc1' not in k}  # Remove keys related to FC layer
            self.model_A.load_state_dict(state_dict_A, strict=False)
            print(f'load model_A weight: {path_model_A}')
        except Exception as e:
            print(f"Failed to load model_A : {e}")
        
        try:

            # Load weights for model B
            state_dict_B = torch.load(path_model_B, map_location='cpu')
            if 'model' in state_dict_B:
                state_dict_B = state_dict_B['model']

            state_dict_B = {k: v for k, v in state_dict_B.items() if 'fc1' not in k}  # Remove keys related to FC layer
            self.model_B.load_state_dict(state_dict_B, strict=False)
            print(f'load model_B weight: {path_model_B}')

        except Exception as e:
            print(f"Failed to load model_B : {e}")
        
            

    def forward(self, x):
        # Forward pass through ResNet-50 models
        feat_A = self.model_A(x)
        feat_B = self.model_B(x)
        
        # Concatenate features from both models
        combined_features = torch.cat((feat_A, feat_B), dim=1)
        
        # MLP for classification
        out = self.head(combined_features)
        
        return out



def resnet50_fusion(**kwargs):
    """
    
    Parameters
    ----------
    **kwargs : num_classes, model1_path, model2_path,
        DESCRIPTION.

    Returns Model fusion
    -------

    """
    num_classes = kwargs.get('num_classes', 1)
    model = ModelFusion(num_classes=num_classes)
    
    path_model_A = kwargs.get('model1_path', '')
    path_model_B = kwargs.get('model2_path', '')

    if (path_model_A and path_model_B) != '':
        model.load_weights(path_model_A, path_model_B)
    else:
        print("Notification: No pretrained weights for model_A or model_B.")
        
    return model


if __name__ == '__main__':
    
    from options.train_options import TrainOptions
    from networks.trainer import Trainer

    from util import get_model
    import torch
    opt = TrainOptions().parse()
    opt.detect_method = 'model_fusion'

    model = Trainer(opt)
    intens = torch.rand(4,3,224,224)
    model.model(intens) 
    #model.save_networks('draf')
    #state_dict = torch.load(r'checkpoints/experiment_name2024_07_09_14_11_23/model_epoch_draf.pth', map_location='cpu')
    #model.model.load_state_dict(state_dict)










