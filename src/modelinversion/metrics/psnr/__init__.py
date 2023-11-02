

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image
import os

def psnr(fake_imgs, real_imgs, combination=False, factor=1.,eps=1e-6):
    """calculate psnr between fake imgs and real imgs

    Args:
        fake_imgs (Tensor): n fake imgs
        real_imgs (Tensor): m real imgs
        batch_size (int, optional): batch size. Defaults to 60.
        combination (bool, optional): 
            if True:
                Combine fake and real each other and calcutate, m * n times
            else:
                fake imgs and real imgs are in pairs, n times
        . Defaults to False.
        factor (float, optional): factor for calculate. Defaults to 1..
    """
    
    if not combination and len(fake_imgs) != len(real_imgs):
        raise RuntimeError('number of fake imgs and real imgs should be the same when combination is False')
    
    
    
    def get_psnr(fake, real):
        mse = ((fake - real) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        return 10 * torch.log10(factor**2 / (mse + eps))
    
    if combination:
        results = []
        for i in range(len(fake_imgs)):
            fake = fake_imgs[i]
            ret = get_psnr(fake, real_imgs).max()
            results.append(ret.item())
        return torch.Tensor(results).to(fake_imgs)
    else:
        return get_psnr(fake_imgs, real_imgs)
        
            
            
def calc_psnr(recovery_img_dir, private_img_dir):
    
    trans = torchvision.transforms.ToTensor()
    
    psnr_all = 0
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
        
        psnr_res = psnr(recovery_imgs, private_imgs, combination=True)
        
        psnr_all += psnr_res.sum().item()
        num += len(psnr_res)
        
    if num == 0:
        raise RuntimeError('no imgs')
        
    res = psnr_all / num
        
    print (f'psnr: {res}')
    
    return res