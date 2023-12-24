import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, ToPILImage, Lambda, Resize

from modelinversion.defense import TLTrainArgs, TLTrainer
from modelinversion.foldermanager import FolderManager
from modelinversion.utils import RandomIdentitySampler
from modelinversion.models import get_model
from development_config import get_dirs




if __name__ == '__main__':
    
    defense_type = 'tl'
    
    dirs = get_dirs(defense_type)
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type)
    
    model_name = 'vgg16'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.0001
    device = 'cuda'
    bido_loss_type = 'hisc'
    batch_size = 64
    
    train_args = TLTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type=defense_type,
        device=device,
        bido_loss_type=bido_loss_type,
        coef_hidden_input=0.05,
        coef_hidden_output=0.5
    )
    
    model = get_model(model_name, dataset_name, device=device, backbone_pretrain=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.5)
    
    trainer = TLTrainer(train_args, folder_manager, model=model, optimizer=optimizer, scheduler=scheduler)
    
    
        
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=Compose([
        RandomHorizontalFlip(p=0.5), ToTensor()
    ]))
    print(trainset[0][0].shape)
    train_sampler = RandomIdentitySampler(trainset, batch_size, 4)
    trainloader = DataLoader(trainset, batch_size, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    trainer.train(trainloader, testloader)
