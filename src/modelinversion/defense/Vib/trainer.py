

from dataclasses import dataclass

import torch.nn.functional as F
from torch import LongTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseTrainer, BaseTrainArgs
from ...models import ModelResult
from ...utils import FolderManager

@dataclass
class VibTrainArgs(BaseTrainArgs):
    beta: float = 1e-2

class VibTrainer(BaseTrainer):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
    
    
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        pred_res = result.result
        mu = result.addition_info['mu']
        std = result.addition_info['std']
        cross_loss = F.cross_entropy(pred_res, labels)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + self.args.beta * info_loss
        return loss
    