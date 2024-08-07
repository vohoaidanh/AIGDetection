import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice, sample
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
import random as rrandom
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
   
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)

    if opt.mode == 'custom':
        return my_binary_dataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                # transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])





import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, num_samples=1):
        # Khởi tạo dataset từ thư mục
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.num_samples = num_samples
        self.indices = list(range(len(self.dataset)))  # Tạo danh sách các chỉ số ảnh
        self.indices_by_class = self._get_indices_by_class()

    def _get_indices_by_class(self):
        # Khởi tạo dictionary để lưu chỉ số của từng lớp
        indices_by_class = {class_idx: [] for class_idx in range(len(self.dataset.classes))}
        
        # Phân loại các chỉ số ảnh theo lớp
        for idx, (path, label) in tqdm(enumerate(self.dataset.samples), total=len(self.dataset.samples), desc="Processing images"):
            indices_by_class[label].append(idx)
        
        return indices_by_class

    def __len__(self):
        # Trả về số lượng batch (chỉ số ảnh)
        return len(self.dataset)  # hoặc số lượng batch tùy ý
    
    def combine_fft(self,main_img, ref_img):
        
        # Thực hiện biến đổi Fourier cho ảnh thứ nhất
        fft_img1 = torch.fft.fft2(main_img)
        #magnitude_img1 = torch.abs(fft_img1)
        phase_img1 = torch.angle(fft_img1)
        
        # Thực hiện biến đổi Fourier cho ảnh thứ hai
        fft_img2 = torch.fft.fft2(ref_img)
        magnitude_img2 = torch.abs(fft_img2)
        #phase_img2 = torch.angle(fft_img2)
        
        
        real = magnitude_img2 * torch.cos(phase_img1)
        imag = magnitude_img2 * torch.sin(phase_img1)
        
        combined_fft = torch.complex(real, imag)
        combined_image = torch.fft.ifft2(combined_fft)

        return torch.abs(combined_image)

    def __getitem__(self, idx):
            # Lấy ảnh và nhãn từ chỉ số idx
            img, label = self.dataset[idx]
            
            if isinstance(img, torch.Tensor):
                img = img
            else:
            # Chuyển đổi ảnh về tensor nếu cần
                if self.transform:
                    img = self.transform(img)
                        
            # Khởi tạo danh sách để lưu ảnh và nhãn
            images = [img]
            #labels = [label]
            
            # Chọn thêm 2 ảnh thuộc lớp 0
            if 0 in self.indices_by_class:
                additional_indices = sample(self.indices_by_class[0], self.num_samples)
            else:
                raise ValueError("Không có ảnh nào thuộc lớp 0 trong dataset")
            
            # Thêm ảnh bổ sung vào danh sách
            for index in additional_indices:
                img2, _ = self.dataset[index]
                if isinstance(img2, torch.Tensor):
                    img2 = img2
                else:
                    if self.transform:
                        img2 = self.transform(img2)
                images.append(img2)


                #labels.append(label)
            rrandom.shuffle(images)
            #images_tensor = torch.cat(images, dim=0)
            images_tensor = self.combine_fft(img, images[-1])
            # Trả về một batch chứa các ảnh và nhãn
            return images_tensor, label



def my_binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    dset = CustomDataset(
            root,
            transforms.Compose([
                rz_func,
                # transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


