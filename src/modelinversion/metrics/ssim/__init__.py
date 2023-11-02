"""This is code based on https://sudomake.ai/inception-score-explained/."""
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import math
import numpy as np
from torch.autograd import Variable
import os
from PIL import Image

def ssim_gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def ssim_create_window(window_size, channel):
    _1D_window = ssim_gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_core(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim(fake_imgs, real_imgs, combination=False, window_size = 11):

    (_, channel, _, _) = fake_imgs.size()
    window = ssim_create_window(window_size, channel)
    
    window = window.to(fake_imgs)
    
    def get_ssim(fake, real):
        return _ssim_core(fake, real, window, window_size, channel, size_average=False)
    
    if combination:
        results = []
        for i in range(len(fake_imgs)):
            fake = fake_imgs[i]
            ret = get_ssim(fake, real_imgs).max() # 改成mean ?
            results.append(ret.item())
        return torch.Tensor(results).to(fake_imgs)
    else:
        return get_ssim(fake_imgs, real_imgs)

def calc_ssim(recovery_img_dir, private_img_dir):
    
    trans = torchvision.transforms.ToTensor()
    
    ssim_all = 0
    num  = 0
    
    for label in os.listdir(recovery_img_dir):
        recovery_label_dir = os.path.join(recovery_img_dir, label)
        private_label_dir = os.path.join(private_img_dir, label)
        if not os.path.exists(private_img_dir):
            continue
        
        def read_imgs(dir_name):
        
            res = []
            for img_name in os.listdir(dir_name):
                img_path = os.path.join(dir_name, img_name)
                try:
                    img = Image.open(img_path)
                except:
                    continue
                res.append(trans(img))
            res = torch.stack(res, dim=0)
            return res
        
        recovery_imgs = read_imgs(recovery_label_dir)
        private_imgs = read_imgs(private_label_dir)
        
        ssim_res = ssim(recovery_imgs, private_imgs, combination=True)
        
        ssim_all += ssim_res.sum().item()
        num += len(ssim_res)
        
    if num == 0:
        raise RuntimeError('no imgs')
        
    res = ssim_all / num
        
    print (f'psnr: {res}')
    
    return res