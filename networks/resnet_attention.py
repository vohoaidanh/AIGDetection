#https://github.com/lucidrains/vit-pytorch
#https://huggingface.co/google/vit-base-patch16-224
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from networks.local_grad import gradient_filter
from networks.resnet_local_grad import resnet50_local_grad
#from networks.resnet import resnet18
# helpers
#######################################
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import models


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Head(nn.Module):
    def __init__(self, in_features, out_features):
        super(Head, self).__init__()
        self.branch0 = nn.Linear(in_features, out_features)
        self.branch1 = nn.Linear(in_features, out_features)
        

    def forward(self, x):
        return 0.5*(self.branch0(x) + self.branch1(x))
    
class ModelCLF(nn.Module):
    def __init__(self, num_classes=1):
        super(ModelCLF, self).__init__()
        self.resnet = resnet50_local_grad(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.head = Head(in_features=512, out_features=num_classes)
    def forward(self, x):
        # Extract features using ResNet
        features = self.resnet(x)
        # Flatten the features for MLP
        features = features.view(features.size(0), -1)
        # Pass features through MLP head
        out = self.head(features)
        return out


class SmallCNN(nn.Module):
    def __init__(self, c, patch_high, patch_width, embedding_size):
        super(SmallCNN, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:6])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_size)  # ResNet-18 outputs 512 features

        

    def forward(self, x):
        # Xử lý input có kích thước [batch_size, seq_leng, c, patch_high, patch_width]
        batch_size, seq_leng, c, patch_high, patch_width = x.size()
        
        # Reshape đầu vào để [batch_size * seq_leng, c, patch_high, patch_width]
        x = x.view(batch_size * seq_leng, c, patch_high, patch_width)
           
        x = self.resnet(x)
        x = self.avgpool(x)
        # Flatten để đưa vào fully connected layer
        x = x.view(batch_size * seq_leng, -1)  # [batch_size * seq_leng, 128 * patch_high * patch_width]
        #x = x.view(x.size(0), -1)

        # Đưa vào fully connected layer để lấy embedding
        x = self.fc(x)
        
        # Reshape để đưa về kích thước [batch_size, seq_leng, embedding_size]
        x = x.view(batch_size, seq_leng, -1)
        
        return x

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
    
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, **kwargs):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.gradient_filter = gradient_filter
        
        self.to_patch_embedding = nn.Sequential(
            #Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            #nn.LayerNorm(patch_dim),
            #nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim),
            #######################
            Rearrange("b c (h p1) (w p2) -> b (h w) c p1 p2", p1 = patch_height, p2 = patch_width),
            SmallCNN(c=3, patch_high=patch_height, patch_width=patch_width, embedding_size=dim)
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device
        x = self.gradient_filter(img)
        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    
def simple_vit(pretrained=False, image_size=224, patch_size=56, num_classes=1,embedding_dim=128, mlp_dim=256, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimpleViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = embedding_dim,
        depth = 6,
        heads = 16,
        mlp_dim = mlp_dim,
        **kwargs
    )
    
    for name, param in model.named_parameters():
        if 'resnet' in name:
            param.requires_grad = False
    

    return model

class PretrainVit(nn.Module):
    def __init__(self, num_classes=1,**kwargs):
        super().__init__()
        #self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(in_features=768, out_features=num_classes)
        self.gradient_filter = gradient_filter
        
    def forward(self,x):
        #x = self.processor(images=x, return_tensors="pt", do_rescale=False)
        x = self.gradient_filter(x)
        x = self.vit(x)
        return x['logits']

def pretrain_vit(**kwargs): 
    model = PretrainVit(**kwargs)
    
    exclude_layers = ['classifier', 'layer.10', 'layer.11']
    #exclude_layers = None
    if exclude_layers is not None:
        for name, param in model.named_parameters():
            if not any(exclude_layer in name for exclude_layer in exclude_layers):
                   param.requires_grad = False
    
    return model
    

if __name__ == '__main__':
    import torch
    #from vit_pytorch import SimpleViT
    
    v = SimpleViT(
        image_size = 224,
        patch_size = 14,
        num_classes = 1,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 512
    )
    
    v = simple_vit(num_classes=1, embedding_dim=256, mlp_dim=256, patch_size=14)
    
    params = v.named_parameters()
    for name, param in v.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    img = torch.randn(2, 3, 224, 224)
    
    preds = v(img) # (1, 1000)
    
# =============================================================================
#     v = pretrain_vit(num_classes=1)
#        
#     preds=preds['logits']
#     
#  
# 
#     
#     att = Attention(dim=784, heads=8,dim_head=64)
#     inputs = torch.rand(2,3,784)
#     out = att(inputs)    
# 
# 
#     
# 
#     module = nn.Sequential(
#         #Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = 56, p2 = 56),
#         #nn.LayerNorm(patch_dim),
#         #nn.Linear(patch_dim, dim),
#         #nn.LayerNorm(dim),
#         SmallCNN(c=3, patch_high=56, patch_width=56, embedding_size=512),
#     )
#         
#     
#     out = module(torch.rand(2,16,3,56,56))
#     
#     c = SmallCNN(3,28,28,128)
#     
#     out = c(torch.rand(4,16,3,28,28))
#     
#     
# 
#     
#     resnet4 = nn.Sequential(*list(resnet.children())[:6])
#     out = resnet4(torch.rand(4,3,224,224))
#     adap = nn.AdaptiveAvgPool2d((1,1))
#     out = adap(out)
#     out = out.view(4,128,28*28)
#     
#     
#     from einops import rearrange
#     
#     # Example input tensor
#     batch_size, channels, height, width = 8, 3, 224, 224
#     image_tensor = torch.randn(batch_size, channels, height, width)
#     
#     # Define patch size
#     patch_height, patch_width = 56, 56
#     
#     # Rearrange operation
#     patches = rearrange(image_tensor, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width)
#     patches = rearrange(image_tensor, "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=patch_height, p2=patch_width)
#     
#     print(patches.shape)  # Output shape: [2, 16, 192]
#     torch.sqrt(torch.tensor(9408))
#     56*56*3
# =============================================================================
model = ModelCLF(num_classes=1)
out = model(torch.rand(13,3,224,224)).detach()




