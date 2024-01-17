from dataclasses import dataclass

import torch
from torch import LongTensor
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseTrainer, BaseTrainArgs
from ...models import ModelResult
from ...foldermanager import FolderManager

@dataclass
class DPTrainArgs(BaseTrainArgs):
    
    noise_multiplier: float = 0.01
    microbatch_size: int = 1
    
    

class DPTrainer(BaseTrainer):
    
    def __init__(self, args: DPTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        self.args: DPTrainArgs
        
        
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        pred_res = result.result
        # pred_res = F.softmax(pred_res, dim=1)
        # return self.criterion(pred_res, labels)
        return F.cross_entropy(pred_res, labels, reduction='none')
    
    
    def before_train(self):
        super().before_train()
        # self.avg_norm = 0
    
    def _update_step(self, loss):
        
        # loss.backward()
        bs = len(loss)
        
        parameters = [param for param in self.model.parameters() if param.requires_grad]
        
        grad = [torch.zeros_like(param) for param in parameters]
        num_microbatch = (bs - 1) // self.args.microbatch_size + 1
        
        max_norm = self.args.clip_grad_norm
        
        # print(len(list(range(0, bs, num_microbatch))))
        # exit()
        for j in range(0, bs, self.args.microbatch_size):
            self.optimizer.zero_grad()
            torch.autograd.backward(torch.mean(loss[j:min(j+self.args.microbatch_size, bs)]), retain_graph=True)
            
            l2norm = 0.
            for param in parameters:
                l2norm += (param.grad * param.grad).sum()
            l2norm = torch.sqrt(l2norm)
            
            # self.avg_norm = self.avg_norm * 0.95 + l2norm * 0.05
            
            
            coef = 1 if max_norm is None else (max_norm / max(max_norm, l2norm.item()))
            grad  = [g + param.grad * coef for param, g in zip(parameters, grad)]
        
        if max_norm is None:
            max_norm = 1.
            
        for param, g in zip(parameters, grad):
            param.grad.data = g
            if self.args.noise_multiplier > 0:
                param.grad.data += torch.cuda.FloatTensor(g.size()).normal_(0, self.args.noise_multiplier * float(max_norm)).to(self.args.device) # torch.randn_like(g) * self.args.noise_multiplier * max_norm
            param.grad.data /= num_microbatch
            
        self.optimizer.step()