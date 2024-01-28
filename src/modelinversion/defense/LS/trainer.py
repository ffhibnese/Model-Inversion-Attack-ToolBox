from dataclasses import dataclass, field

import torch
from torch import LongTensor
from torch.nn import Module, MaxPool2d, Sequential
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F

from ...models import ModelResult
from ...models.base import BaseTargetModel
from ...utils import traverse_module, OutputHook, BaseHook
from ...foldermanager import FolderManager
from ..base import BaseTrainArgs, BaseTrainer

@dataclass
class LSTrainArgs(BaseTrainArgs):
    
    coef_label_smoothing: float = 0.1
    
    
class LSTrainer(BaseTrainer):
    
    def __init__(self, args: LSTrainArgs, folder_manager: FolderManager, model: BaseTargetModel, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, lr_scheduler, **kwargs)
        
    def _neg_label_smoothing(self, inputs, labels):
        ls = self.args.coef_label_smoothing
        confidence = 1.0 - ls
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + ls * smooth_loss
        return torch.mean(loss, dim=0).sum()
        
        
    def calc_loss(self, inputs: torch.Tensor, result: ModelResult, labels: LongTensor):
        res = result.result
        bs = len(inputs)
        
        return self._neg_label_smoothing(res, labels)