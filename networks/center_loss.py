import torch
import torch.nn as nn


__all__ = ['CycleLoss', 'CenterLoss']

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(mat1=x, mat2=self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    

    
class CycleLoss(nn.Module):

    def __init__(self,num_classes = 2, margin = 1.0, feat_dim=2, use_gpu=True):
        super(CycleLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(1, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):

        x0 = x[labels==0]
        x1 = x[labels==1]
        
        if len(x0) == 0:
            d0=torch.tensor(0.0)
        else:
            d0 = torch.square(x0 - self.centers)
            mask = d0 < 0.02
            d0[mask] = 0.0
            d0 = d0.clamp(min=1e-12, max=1e+12).sum(dim=1)
            d0 = torch.mean(d0)
        
        if len(x1) == 0:
            d1 = torch.tensor(0.0)
        else:
            d1 = torch.square(x1 - self.centers)
            d1 = d1.clamp(min=1e-12, max=1e+12).sum(dim=1)
            d1 = torch.log(self.margin/torch.mean(d1))
        
        loss = d0 + d1.clamp(min=0.0)
        
        return loss
    
    
if __name__ == '__main__':
    
    cycle_loss = CycleLoss(num_classes=2, feat_dim=3, use_gpu=False)

    
    center_loss = CenterLoss(num_classes=2, feat_dim=3, use_gpu=False)
    
    x1 = torch.tensor([[1.,2., 3.],[4.,1., 2.]]).view(2,-1)
    y1 = torch.tensor([0,0])
    
    x2 = torch.tensor([2., 1., 3.]).view(1,-1)
    y2 = torch.tensor([1])
    
    c = center_loss.centers
    
    x = torch.cat((x1,x2), dim=0)
    y = torch.cat((y1,y2), dim = 0)
    
    c1 = center_loss(x1, y1)
    c2 = center_loss(x2, y2)
    (c1*2 + c2)/3
    
    center_loss(x,y)
    
    cycle_loss(x1,y1)
    
    with torch.no_grad():
        print(torch.mean(torch.sum(torch.square(x1 - c[0]), dim=1))/2)
        print(torch.mean(torch.sum(torch.square(x2 - c[1]), dim=1))/1)
    
    
    
    
    distmat = torch.pow(x,2).sum(dim=1, keepdim=True).expand(4,2) + \
        torch.pow(c, 2).sum(dim=1, keepdim=True).expand(2, 4).t()
    
    distmat.addmm_(mat1=x, mat2=c.t(), beta=1, alpha=-2)
    
  
    (torch.mean(torch.sum(torch.square(x1 - c[0]), dim=1)) + torch.mean(torch.sum(torch.square(x2 - c[1]), dim=1)))/3
      
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
