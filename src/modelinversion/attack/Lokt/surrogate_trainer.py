import os
from typing import Callable
from dataclasses import field, dataclass

import kornia
import torch
from torch import nn
from torch._C import LongTensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modelinversion.models import ModelResult

from ...foldermanager import FolderManager
from ...models import get_model, BaseTargetModel
from ...trainer import BaseTrainArgs, BaseTrainer
from ..PLGMI.code.m_cgan import ResNetGenerator
from .models.discri import SNResNetConditionalDiscriminator

class LoktSurrogateTrainArgs(BaseTrainArgs):
    augment: Callable = field(default_factory=
        kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        )
    )

class LoktSurrogateTrainer(BaseTrainer):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: FolderManager, model: BaseTargetModel, generator, discriminator, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, lr_scheduler, **kwargs)
        
        self.G = generator
        self.D = discriminator
        
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        return super().calc_loss(inputs, result, labels)