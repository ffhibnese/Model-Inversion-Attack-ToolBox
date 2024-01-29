import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, ToPILImage, Lambda, Resize, CenterCrop, RandomResizedCrop, ColorJitter, Normalize
from torchvision import transforms as tvtrans

from development_config import get_dirs
from modelinversion.defense import *
from modelinversion.models import get_model
from modelinversion.foldermanager import FolderManager
from modelinversion.utils import RandomIdentitySampler
import math

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    defense_type = 'ls'
    
    model_name = 'resnet152'
    
    dirs = get_dirs(f'{defense_type}_{model_name}')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type)
    
    dataset_name = 'hdceleba'
    epoch_num = 100
    lr = 0.001
    device = 'cuda'
    batch_size = 128
    resolution = 224
    
    coef_label_smoothing = 0.1
    
    args = LSTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='no_defense',
        tqdm_strategy=TqdmStrategy.ITER,
        device=device,
        coef_label_smoothing=coef_label_smoothing
    )
    
    model = get_model(model_name, dataset_name, device, backbone_pretrain=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    
    trainer = LSTrainer(args, folder_manager, model, optimizer, scheduler)
    
    train_transform = Compose([
        Resize((resolution, resolution), antialias=True),
        ToTensor(),
        RandomResizedCrop(size=(resolution, resolution), scale=[0.85, 1], ratio=[1,1], antialias=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        RandomHorizontalFlip(0.5),
        Normalize(0.5, 0.5)
    ])
    
    trainset = ImageFolder('./dataset/hdceleba/split/private/train', transform=Compose([
        RandomHorizontalFlip(p=0.5), ToTensor()
    ]))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    test_transform = Compose([
        Resize((resolution, resolution), antialias=True),
        ToTensor(),
        Normalize(0.5, 0.5)
    ])
    
    testset = ImageFolder('./dataset/hdceleba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    trainer.train(trainloader, testloader)
