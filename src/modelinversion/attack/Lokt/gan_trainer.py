
import os
import time
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import kornia
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tv_trans
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

from modelinversion.metrics.base import DataLoader


from ..base import BaseGANTrainArgs, BaseGANTrainer
from ...models import *
from ...utils import walk_imgs, print_as_yaml
# from ..code.m_cgan import ResNetGenerator, SNResNetProjectionDiscriminator
from ..PLGMI.code.m_cgan import ResNetGenerator
from .models.discri import SNResNetConditionalDiscriminator
         
@dataclass
class LoktGANTrainArgs(BaseGANTrainArgs):
    top_n: int = 30
    target_name: str = 'vgg16'
    # num_classes: int = 1000
    # augment: Callable = field(default_factory=lambda: kornia.augmentation.container.ImageSequential(
    #     kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    #     kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
    #     kornia.augmentation.RandomHorizontalFlip(),
    #     kornia.augmentation.RandomRotation(5),
    # ))
    inv_loss_start_iter: int = 100
    
    coef_inv_loss: float = 0.2
    lr: float= 0.0002
    beta1: float = 0.0
    beta2: float = 0.9
    z_dim = 128
    gen_distribution = 'normal'
            
class LoktGANTrainer(BaseGANTrainer):
    
    def __init__(self, args: LoktGANTrainArgs, folder_manager, **kwargs) -> None:
        super().__init__(args, folder_manager, **kwargs)
        self.args: LoktGANTrainArgs
        self.num_classes = NUM_CLASSES[args.dataset_name]
        
        self.src_dataset_dir = os.path.join(folder_manager.config.dataset_dir, args.dataset_name, 'split', 'public')
        # self.dst_dataset_dir = os.path.join(folder_manager.config.cache_dir, args.dataset_name, args.target_name)
        
    def get_tag(self) -> str:
        args = self.args
        return f'lokt_{args.dataset_name}_{args.target_name}'
    
    def get_method_name(self) -> str:
        return 'lokt'

    def get_trainloader(self) -> DataLoader:
        dataset = ImageFolder(self.src_dataset_dir, transform=tv_trans.ToTensor())
        dataloader = DataLoader(dataset, self.args.batch_size, shuffle=True)
        return dataloader
        
    def prepare_training(self):
        # return "maomao"
        args = self.args
        self.G = ResNetGenerator(dim_z=args.z_dim, num_classes=self.num_classes, distribution=args.gen_distribution).to(args.device)
        self.D = SNResNetConditionalDiscriminator(num_classes=self.num_classes).to(args.device)
        self.T = get_model(args.target_name, args.dataset_name, device=args.device, backbone_pretrain=False, defense_type=args.defense_type)
        self.folder_manager.load_target_model_state_dict(self.T, args.dataset_name, args.target_name, device=args.device, defense_type=args.defense_type)
        self.T.eval()

        
        self.optim_G = torch.optim.Adam(self.G.parameters(), args.lr, (args.beta1, args.beta2))
        self.optim_D = torch.optim.Adam(self.D.parameters(), args.lr, (args.beta1, args.beta2))
    
    def _sample(self, batch_size):
        args = self.args
        z = torch.randn((batch_size, args.z_dim), device=args.device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=args.device)
        fake = self.G(z, y)
        return z, y, fake
    
    def _max_margin_loss(self, out, iden):
        real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
        tmp1 = torch.argsort(out, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
        margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

        return (-1 * real).mean() + margin.mean()
    
    def train_gen_step(self, batch):
        args = self.args
        bs = len(batch[0])
        _, labels, fake = self._sample(bs)
        dis_fake, dis_class = self.D(fake)
        
        dis_loss = F.binary_cross_entropy(dis_fake, torch.ones_like(dis_fake))
        inv_loss = 0
        if self.iteration > args.inv_loss_start_iter:
            inv_loss = F.cross_entropy(dis_class, labels)
        loss = dis_loss + args.coef_inv_loss * inv_loss
        
        super().loss_update(loss, self.optim_G)
        
        return OrderedDict(
            dis_loss = dis_loss.item(),
            inv_loss = inv_loss.item(),
            total_loss = loss.item()
        )
        
    def train_dis_step(self, batch):
        args = self.args
        bs = len(batch[0])
        
        _, labels, fake = self._sample(bs)
        dis_fake, dis_fake_class = self.D(fake)
        dis_real, _ = self.D(batch)
        
        loss_fake = F.binary_cross_entropy(dis_fake, torch.zeros_like(dis_fake))
        loss_real = F.binary_cross_entropy(dis_real, torch.ones_like(dis_real))
        
        dis_loss = loss_fake + loss_real
        # exit()
        
        loss = dis_fake + dis_real
           
           
        super().loss_update(loss, self.optim_D)
        
        # return {
        #     'fake loss': dis_fake.item(),
        #     'real loss': dis_real.item(),
        #     'total loss': loss.item()
        # }
        return OrderedDict(
            fake_loss = dis_fake.item(),
            real_loss = dis_real.item(),
            total_loss = loss.item()
        )