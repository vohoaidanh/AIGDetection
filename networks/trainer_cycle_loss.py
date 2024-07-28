import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from util import get_model
from networks.center_loss import CycleLoss as CenterLoss
class Trainer(BaseModel):
    alpha = 1.0
    target_center_loss = None

    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            #self.model = resnet50(pretrained=False, num_classes=1)
            self.model = get_model(opt)

        if not self.isTrain or opt.continue_train:
            #self.model = resnet50(num_classes=1)
            self.model = get_model(opt)

        if self.isTrain:
            #self.loss_fn = nn.BCEWithLogitsLoss()
            
            self.center_loss = CenterLoss(num_classes=2,margin=1.0, feat_dim=512, use_gpu=True)
            self.loss_fn = getattr(nn, opt.loss_fn)()

            # initialize optimizers
            if opt.optim == 'adam':
                model_params = filter(lambda p: p.requires_grad, self.model.parameters())
                params = list(model_params) + list(self.center_loss.parameters())
                self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                model_params = filter(lambda p: p.requires_grad, self.model.parameters())
                params = list(model_params) + list(self.center_loss.parameters())
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
                
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
            
        if torch.cuda.is_available() :
            self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        #Trainer.alpha = Trainer.alpha*0.8
        #if Trainer.alpha < 1e-3:
        #    Trainer.alpha=0.0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.features, self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        #self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.center_loss_value = self.center_loss(self.features, self.label)
                    
        self.loss = self.center_loss_value * Trainer.alpha + 1.0*self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        for param in self.center_loss.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            lr_cent = self.lr*20.0
            param.grad.data *= (lr_cent / (Trainer.alpha * self.lr))
          
        #self.optimizer.zero_grad()
        #self.loss.backward()
        self.optimizer.step()




