import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import collections
from torchvision.transforms import ToTensor
from tqdm import tqdm

def generate_private_feats(eval_model, img_dir, save_dir, transforms=None, batch_size=60, device='cpu', exist_ignore=False):
    
    print(f'generate feats form {img_dir} to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    
    if len(os.listdir(save_dir)) and exist_ignore > 2:
        return
    
    results = collections.defaultdict(list)
    dataset = ImageFolder(img_dir, transform=ToTensor())
    data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for imgs, labels in tqdm(data_loaders):
        if transforms is not None:
            imgs = transforms(imgs)
        
        imgs = imgs.to(device)
        
        feat = eval_model(imgs).feat[-1].cpu()
        for i in range(len(labels)):
            # save_dir = os.path.join(save_dir, f'{labels[i].item()}')
            # os.makedirs(save_dir)
            label = labels[i].item()
            results[label].append(feat[i].detach().cpu().unsqueeze(0).numpy())
            
    for label, feats in results.items():
        feats = np.concatenate(feats, axis=0)
        np.save(os.path.join(save_dir, f'{label}.npy'), feats)

def calc_knn(fake_feat_dir, private_feat_dir):
    fake_feat_files = os.listdir(fake_feat_dir)
    
    total_knn = 0
    total_num = 0
    
    print(f'calculate knn\n fake from {fake_feat_dir}\n private from {private_feat_dir}')
    
    for fake_feat_file in fake_feat_files:
        fake_path = os.path.join(fake_feat_dir, fake_feat_file)
        private_path = os.path.join(private_feat_dir, fake_feat_file)
        if not os.path.exists(private_path):
            continue
        
        # (N_f, 1, dim)
        fake_feat = np.load(fake_path)[:, None, :]
        # (1, N_p, dim)
        private_feat = np.load(private_path)[None, :, :]
        # (N_f, N_p)
        diff = ((fake_feat - private_feat) ** 2).sum(axis=-1)
        
        knns = np.min(diff, axis=1)
        total_knn += knns.sum()
        total_num += len(knns)
    if total_num == 0:
        raise RuntimeError('NO feat file for fake or private')
    
    return total_knn / total_num
