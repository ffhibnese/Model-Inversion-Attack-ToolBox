
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
    
    target_name: str = 'vgg16'
    class_loss_start_iter: int = 1000
    
    coef_class_loss: float = 1.5
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
    
    def train_gen_step(self, batch):
        args = self.args
        bs = len(batch[0])
        _, labels, fake = self._sample(bs)
        dis_fake, dis_class = self.D(fake)
        
        dis_loss = F.binary_cross_entropy(dis_fake, torch.ones_like(dis_fake))
        class_loss = 0
        if self.iteration > args.class_loss_start_iter:
            class_loss = F.cross_entropy(dis_class, labels)
        loss = dis_loss + args.coef_class_loss * class_loss
        
        super().loss_update(loss, self.optim_G)
        
        return OrderedDict(
            dis_loss = dis_loss.item(),
            class_loss = class_loss,
            total_loss = loss.item()
        )
        
    def train_dis_step(self, batch):
        args = self.args
        
        batch = batch[0].to(args.device)
        bs = len(batch)
        
        _, labels, fake = self._sample(bs)
        dis_fake, dis_fake_class = self.D(fake)
        dis_real, _ = self.D(batch)
        
        loss_fake = F.binary_cross_entropy(dis_fake, torch.zeros_like(dis_fake))
        loss_real = F.binary_cross_entropy(dis_real, torch.ones_like(dis_real))
        
        class_loss = 0
        dis_acc = 0
        gen_acc = 0
        if self.iteration > args.class_loss_start_iter:
            with torch.no_grad():
                y_prob_pred = self.T(fake).result
                y_pred = torch.argmax(y_prob_pred, dim=-1)
                dis_acc = (y_pred == torch.argmax(dis_fake_class)).float().mean()
                gen_acc = (y_pred == labels).float().mean()
                
            class_loss = F.cross_entropy(dis_fake_class, y_pred)
        
        loss = loss_fake + loss_real + args.coef_class_loss * class_loss
           
           
        super().loss_update(loss, self.optim_D)
        
        return OrderedDict(
            fake_loss = loss_fake.item(),
            real_loss = loss_real.item(),
            class_loss = class_loss,
            total_loss = loss.item(),
            dis_acc = dis_acc,
            gen_acc = gen_acc
        )