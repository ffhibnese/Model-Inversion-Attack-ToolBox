

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np

def psnr(fake_imgs, real_imgs, combination=False, factor=1.,eps=1e-6):
    """calculate psnr between fake imgs and real imgs

    Args:
        fake_imgs (Tensor): n fake imgs
        real_imgs (Tensor): m fake imgs
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
        
            
            
def calc_psnr(recovery_img_path, private_img_path, batch_size=64, device='cpu'):
    
    recover_loader = DataLoader(
        torchvision.datasets.ImageFolder(
            recovery_img_path,
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ), batch_size, 
    )
    
    private_loader = DataLoader(
        torchvision.datasets.ImageFolder(
            private_img_path,
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ), batch_size, 
    )
    
    final_results = []
    for recover_imgs, _ in recover_loader:
        result = None
        for private_imgs, _ in private_loader:
            batch_psnr = psnr(recover_imgs, private_imgs, combination=False)
            if result is None:
                result = batch_psnr
            else:
                result = torch.where(result > batch_psnr, result, batch_psnr)
        final_results += result.detach().cpu().numpy().tolist()
        
    res = np.mean(final_results)
    print (f'psnr: {res}')
    
    return res