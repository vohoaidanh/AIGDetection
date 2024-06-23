import sys
import matplotlib.pyplot as plt
import torch
#import torch.nn.functional as F
#import torchvision.transforms.functional as TF
#from PIL import Image
#import cv2
from util import get_model
from options import TestOptions
from data import create_dataloader
from tqdm import tqdm
import numpy as np

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


def create_gaussian_matrix(size, sigma):
    # Create coordinate grids
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    y_grid, x_grid = torch.meshgrid(y, x)

    # Calculate Gaussian matrix
    gauss_matrix = torch.exp(-((x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2)))
    return gauss_matrix

def show_mask(feature, size = (16*28,32*28), label='label_0'):       
    c,h,w = feature.shape
    height, width = size
    assert height*width == c*h*w , 'Size does not match'
    mask = feature.view(height//h, width//w, h, w)
    mask = mask.permute(0,2,1,3)
    mask = mask.reshape(height,width)
    plt.imshow(mask)
    plt.title(label)
    plt.show()
    return mask

class Mean:
    def __init__(self):
        self.sum = None
        self.count = 0

    def add(self, tensor):
        self.count += 1
        
        if self.sum is None:
            # Initialize sum with the first tensor
            self.sum = tensor.float().mean(dim=0)
        else:
            # Add the new tensor to the running sum
            self.sum += tensor.float().mean(dim=0)
    
    @property 
    def average(self):
        return self.sum/self.count
    
    def reset(self):
        self.__init__()
    
if __name__ == '__main__':
    

    opt = TestOptions().parse(print_options=False)
    opt.model_path = r'weights/Gradient-1class-car2024_06_20_model_eopch_7_best.pth'
    opt.dataroot = r"D:\Downloads\dataset\CNN_synth\gaugan".replace('\\', '/')
    opt.classes  = [] #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True
    opt.no_crop   = False
    opt.detect_method = 'local_grad'
    opt.batch_size = 16
    opt.num_threads = 0
    opt.isTrain=True
    opt.register_hook = True
    data_loader = create_dataloader(opt)
    
# =============================================================================
#     MASK = None
#     data0 = torch.load(r'weights/progan_feature_label_0.pth')
#     data1 = torch.load(r'weights/progan_feature_label_1.pth')
#     diff = data0['mean_layer2']  > data1['mean_layer2']
#     MASK = diff.float()
# =============================================================================
        
    data0 = torch.load(r'weights/progan_feature_label_0.pth')
    data1 = torch.load(r'weights/progan_feature_label_1.pth')
    a = (data0['mean_layer2'] > 1.0*data0['mean_layer2'].mean()).float()
    b = (data1['mean_layer2'] > 1.5*data1['mean_layer2'].mean()).float()
    MASK = (a + b).float()
    print(f'Model_path {opt.model_path}')
    model = get_model(opt)
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    
    target_layers = [model.layer2[3].relu]
    #cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = []

    model.eval()
    # Define a hook function to save the outputs
    outputs = {}

    def hook_fn(module, input, output):
        layer_name = module_name_map[module]
        if output.shape[1:] in [(256,56,56), (512,28,28)]:
            if layer_name not in outputs:
                outputs[layer_name] = []
            #outputs[layer_name].extend(output)
            if MASK is not None:
                outputs[layer_name] = output
                return output * MASK.to(output.device)
            else:
                outputs[layer_name] = output
        
    module_name_map = {}
    def register_hook(layer_name, module):
        module_name_map[module] = layer_name
        hook = module.register_forward_hook(hook_fn)
        return hook
    
    # Access the specific submodules
    #hook1 = register_hook('layer1.2.relu', model.layer1[2].relu)
    hook2 = register_hook('layer2.3.relu', model.layer2[3].relu)
    mean_layer1 = Mean()
    mean_layer2 = Mean()        

    with torch.no_grad():
        for LABEL in [0,1]:
            y_true, y_pred = [], [] ; y_pred_no_active=[]
            count = 0
            mean_layer1.reset()
            mean_layer2.reset()
            for data, label in tqdm(data_loader):
                count += 1
                if count > 100:
                    break
                mask = label==LABEL 
                if sum(mask) == 0:
                    continue
                in_tens = data[mask].to(device)
                y_pred.extend(model(in_tens).detach().sigmoid().flatten().tolist())
                y_pred_no_active.extend(model(in_tens).detach().flatten().tolist())
                
                y_true.extend(label[mask].flatten().tolist())
                
                #mean_layer1.add(outputs['layer1.2.relu'])
                mean_layer2.add(outputs['layer2.3.relu'])
    
            data = {
                'y_true': y_true,
                'y_pre':y_pred,
                'y_pred_no_active': y_pred_no_active,
                #'mean_layer1': mean_layer1.average,
                'mean_layer2': mean_layer2.average,
                }
            if opt.register_hook:
                save_file_path = f'weights/CNN_synth_gaugan_feature_label_{LABEL}_mask.pth'
                torch.save(data,save_file_path)
            
    #hook1.remove()
    hook2.remove()
    

data0 = torch.load(r'weights/CNN_synth_gaugan_feature_label_0_mask.pth')
data1 = torch.load(r'weights/CNN_synth_gaugan_feature_label_1_mask.pth')
#a = (data0['mean_layer2'] > 1.5*data0['mean_layer2'].mean()).float()
#b = (data1['mean_layer2'] > 1.5*data1['mean_layer2'].mean()).float()
#MASK = (a + b).float()
#show_mask(data0['mean_layer2'],label='0')
show_mask(data['mean_layer2'],label='1')   
show_mask(MASK.float(),label='progan_diff', size = (16*28,32*28))


ypre_0 = np.asarray(data0['y_pred_no_active'])
ypre_1 = np.asarray(data1['y_pred_no_active'])

r_acc =  np.sum(np.asarray(data0['y_pre'])<0.5)/len(np.asarray(data0['y_pre']))
f_acc = np.sum(np.asarray(data1['y_pre'])>=0.5)/len(np.asarray(data1['y_pre']))
print(f'r_acc: {r_acc}, f_acc: {f_acc}')
    
  

plt.hist(ypre_0, bins=50, alpha=0.9, color='green', edgecolor='black',label='real')
plt.hist(ypre_1, bins=50, alpha=0.3, color='red', edgecolor='black',label='fake')
plt.legend()

































