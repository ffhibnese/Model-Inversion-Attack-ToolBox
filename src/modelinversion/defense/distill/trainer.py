
from torch import LongTensor
import torch

from modelinversion.models import ModelResult
from ..base import *
from ...models import ModelResult
import torch.nn.functional as F
from ...foldermanager import FolderManager
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class DistillTrainer(BaseTrainer):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, teacher_model: Module = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        assert teacher_model is not None
        self.teacher = teacher_model.to(args.device)
        self.teacher.eval()
        
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        return 0
        
    
    def _train_step(self, inputs, labels):
        self.before_train_step()
        
        result = self.model(inputs).result
        
        pred_res = result
        teacher_res = self.teacher(inputs).result
        loss = F.kl_div(
            F.log_softmax(pred_res, dim=-1),
            F.softmax(teacher_res, dim=-1),
            reduction='sum'
        )
        
        acc = (torch.argmax(pred_res, dim=-1) == torch.argmax(teacher_res, dim=-1)).float().mean()
        
        self._update_step(loss)
        
        # return TrainStepResult(loss.mean().item(), acc.item())
        return {
            'loss': loss.mean().item(),
            'acc': acc.item()
        }
