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
from modelinversion.attack.base import BaseGANTrainArgs
from modelinversion.foldermanager import FolderManager

from modelinversion.metrics.base import DataLoader, FolderManager


from ..base import BaseGANTrainArgs, BaseGANTrainer
from ...models import *
from .code.generator import Generator
from .code.discri import DGWGAN

@dataclass
class GmiGANTrainArgs(BaseGANTrainArgs):
    
    lr: float= 0.0002
    beta1: float = 0.5
    beta2: float = 0.9
    z_dim = 100
    
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
    
class GmiGANTrainer(BaseGANTrainer):
    
    def __init__(self, args: BaseGANTrainArgs, folder_manager: FolderManager, **kwargs) -> None:
        super().__init__(args, folder_manager, **kwargs)
        self.args: GmiGANTrainArgs
        
    def get_tag(self) -> str:
        args = self.args
        return f'gmi_high_{args.dataset_name}'
    
    def get_method_name(self) -> str:
        return 'GMI_high'
    
    def get_trainloader(self) -> DataLoader:
        ds = ImageFolder('dataset/metfaces/split/public', transform=tv_trans.Compose([
            tv_trans.ToTensor(),
            tv_trans.CenterCrop((800,800)),
            tv_trans.Resize((256,256))
        ]))
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=True)
        
    def prepare_training(self):
        args = self.args
        self.G = (Generator(args.z_dim)).to(args.device)
        self.D = (DGWGAN()).to(args.device)
        
        self.optim_G = torch.optim.Adam(self.G.parameters(), args.lr, (args.beta1, args.beta2))
        self.optim_D = torch.optim.Adam(self.D.parameters(), args.lr, (args.beta1, args.beta2))
        
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor):
        shape = [len(real)] + [1] * (real.ndim - 1)
        alpha = torch.randn(shape).to(self.args.device)
        z = real + alpha * (fake - real)
        z = z.detach().clone()
        z.requires_grad_(True)
        
        o = self.D(z)
        g = torch.autograd.grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph=True)[0].view(len(real), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
        return gp
        
    def train_dis_step(self, batch) -> OrderedDict:
        args = self.args
        
        # freeze(self.G)
        # unfreeze(self.D)
        
        batch = batch[0].to(args.device)
        bs = len(batch)
        z = torch.randn(bs, args.z_dim).to(args.device)
        
        fake = self.G(z)
        dis_fake = self.D(fake)
        dis_real = self.D(batch)
        wd = dis_fake.mean() - dis_real.mean()
        gp = 0 # self._gradient_penalty(batch, fake)
        loss = - wd + gp * 10.
        
        super().loss_update(loss, self.optim_D)
        
        return OrderedDict(
            wasserstein_loss = wd,
            gradient_penalty = gp,
            loss = loss
        )
        
    # def before_gen_train_step(self):
    #     # return super().before_gen_train_step()
    #     z = torch.randn(2, self.args.z_dim).to(self.args.device)
    #     while 1:
    #         fake = self.G(z)
    #         df = self.D(fake)
    
    def train_gen_step(self, batch) -> OrderedDict:
        # return {}
        
        # freeze(self.D)
        # unfreeze(self.G)
        args = self.args
        bs = len(batch[0])
        z = torch.randn(bs, args.z_dim).to(args.device)
        
        # print ('bbb')
        # while 1:
        #     pass
        
        fake = self.G(z)
        # print(fake.shape)
        # exit()
        dis_fake = self.D(fake)
        loss = - dis_fake.mean()
        # print ('aaa')
        # while 1:
        #     pass
        super().loss_update(loss, self.optim_G)
        
        return OrderedDict(loss = loss)