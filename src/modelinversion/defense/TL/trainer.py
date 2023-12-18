

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import LongTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseTrainer, BaseTrainArgs
from ..no_defense.trainer import RegTrainer
from ...models import ModelResult
from ...utils import FolderManager

@dataclass
class TLTrainArgs(BaseTrainArgs):
    pass

class TLTrainer_stage1(RegTrainer):
    
    def __init__(self, args: TLTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        
    def save_state_dict(self):
        # return super().save_state_dict()
        save_path = os.path.join(self.folder_manager.config.cache_dir, f'{self.args.model_name}_{self.args.dataset_name}.pt')
        torch.save({'state_dict': self.model.state_dict()}, save_path)
        

class TLTrainer_stage2(RegTrainer):
    
    def __init__(self, args: TLTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        load_path = os.path.join(self.folder_manager.config.cache_dir, f'{self.args.model_name}_{self.args.dataset_name}.pt')
        state_dict = torch.load(load_path, map_location=self.args.device)['state_dict']
        self.model.load_state_dict(state_dict)
    
    
    def before_train(self):
        self.model.freeze_front_layers()
    