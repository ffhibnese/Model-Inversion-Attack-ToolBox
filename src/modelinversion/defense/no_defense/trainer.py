
from torch import LongTensor
from ..base import BaseTrainer, BaseTrainArgs
from ...models import ModelResult
import torch.nn.functional as F
from ...utils import FolderManager
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class RegTrainer(BaseTrainer):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        
        
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        pred_res = result.result
        # return self.criterion(pred_res, labels)
        return F.cross_entropy(pred_res, labels)