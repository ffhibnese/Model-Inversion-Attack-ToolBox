import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import LongTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseTrainer, BaseTrainArgs
from ..BiDO.trainer import BiDOTrainer, BiDOTrainArgs
from ...models import ModelResult
from ...foldermanager import FolderManager


@dataclass
class TLTrainArgs(BiDOTrainArgs):
    pass


class TLTrainer(BiDOTrainer):

    def __init__(
        self,
        args: TLTrainArgs,
        folder_manager: FolderManager,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        **kwargs
    ) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)

    def before_train_step(self):
        super().before_train_step()
        self.model.freeze_front_layers()
