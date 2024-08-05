#https://github.com/lucidrains/vit-pytorch
#https://huggingface.co/google/vit-base-patch16-224
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from networks.local_grad import gradient_filter

#from networks.resnet import resnet18
# helpers
#######################################
from transformers import ViTImageProcessor, ViTForImageClassification


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.gradient_filter = gradient_filter
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
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
    
    
def simple_vit(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimpleViT(
        image_size = 224,
        patch_size = 28,
        num_classes = 1,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 512
    )

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
    
    #exclude_layers = ['classifier', 'layer.10', 'layer.11']
    exclude_layers = []
    for name, param in model.named_parameters():
        if not any(exclude_layer in name for exclude_layer in exclude_layers):
               param.requires_grad = False
    
    return model
    

if __name__ == '__main__':
    import torch
    #from vit_pytorch import SimpleViT
    
    v = SimpleViT(
        image_size = 224,
        patch_size = 28,
        num_classes = 1,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 512
    )
    
    img = torch.randn(2, 3, 224, 224)
    
    preds = v(img) # (1, 1000)
    
    v = pretrain_vit(num_classes=1)
       
    preds=preds['logits']
    
 

# =============================================================================
#     
#     att = Attention(dim=784, heads=8,dim_head=64)
#     inputs = torch.rand(2,3,784)
#     out = att(inputs)    
# 
# class SmallCNN(nn.Module):
#     def __init__(self, c, patch_high, patch_width, embedding_size):
#         super(SmallCNN, self).__init__()
#         
#         # Các lớp CNN để trích xuất đặc trưng
#         inplanes=64
#         planes = 3
#         stride = 1
#         self.expansion=1
#         
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.relu = nn.ReLU(inplace=True)
#         #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 
#         
#         # Kích thước đầu vào của fully connected layer sau các lớp conv
#         self.fc_input_dim = 128
# 
#         # Fully connected layer để tạo ra embedding
#         self.fc = nn.Linear(self.fc_input_dim, embedding_size)
#         
# 
#     def forward(self, x):
#         # Xử lý input có kích thước [batch_size, seq_leng, c, patch_high, patch_width]
#         batch_size, seq_leng, c, patch_high, patch_width = x.size()
#         
#         # Reshape đầu vào để [batch_size * seq_leng, c, patch_high, patch_width]
#         x = x.view(batch_size * seq_leng, c, patch_high, patch_width)
#         
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
# 
#         out = self.conv2(out)
#         out = self.bn2(out)
# 
# 
#         out = self.relu(out)
#        
#         x = self.avgpool(out)
#         # Flatten để đưa vào fully connected layer
#         x = x.view(batch_size * seq_leng, -1)  # [batch_size * seq_leng, 128 * patch_high * patch_width]
#         #x = x.view(x.size(0), -1)
# 
#         # Đưa vào fully connected layer để lấy embedding
#         x = self.fc(x)
#         
#         # Reshape để đưa về kích thước [batch_size, seq_leng, embedding_size]
#         x = x.view(batch_size, seq_leng, -1)
#         
#         return x
#     
#     
#     patch_dim=3*56*56
#     dim=512
#     module = nn.Sequential(
#         Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = 56, p2 = 56),
#         nn.LayerNorm(patch_dim),
#         nn.Linear(patch_dim, dim),
#         nn.LayerNorm(dim),
#     )
#         
#     
#     out = module(torch.rand(2,3,224,224))
#     
#     c = SmallCNN(3,28,28,512)
#     
#     out = c(torch.rand(4,16,3,28,28))
#     
#     
# 
# =============================================================================

