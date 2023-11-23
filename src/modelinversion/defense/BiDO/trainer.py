from torch.nn import Module, MaxPool2d, Sequential
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
from ...models import ModelResult
from ...utils import FolderManager, traverse_module, OutputHook, BaseHook
from ..base import BaseTrainArgs, BaseTrainer
from torch import LongTensor
import torch
from dataclasses import dataclass, field
from ...models.get_models import NUM_CLASSES
from .kernel import hsic_objective, coco_objective

@dataclass
class BiDOTrainArgs(BaseTrainArgs):
    
    kernel_type: str = field(default='linear', metadata={'help': 'kernel type: linear, gaussian, IMQ'})
    
    bido_loss_type: str = field(default='hisc', metadata={'help': 'loss type: hisc, coco'})
    
    coef_hidden_input: float = field(default=0.05, metadata={'help': 'coef of loss between hidden and input'})
    coef_hidden_output: float = field(default=0.5, metadata={'help': 'coef of loss between hidden and output'})
    
    
class BiDOTrainer(BaseTrainer):
    
    def __init__(self, args: BiDOTrainArgs, folder_manager: FolderManager, model: Module, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, lr_scheduler, **kwargs)
        
        self.hiddens_hooks: list[BaseHook] = []
        
        
        if self.args.bido_loss_type == 'hisc':
            self.objective_fn = hsic_objective
        elif self.args.bido_loss_type == 'coco':
            self.objective_fn = coco_objective
        else:
            raise RuntimeError(f'loss type `{self.args.bido_loss_type}` is not supported, valid loss types: `hisc` and `coco`')
            
        
    def _to_onehot(self, y, num_classes):
        """ 1-hot encodes a tensor """
        # return torch.squeeze(torch.eye(num_classes)[y.cpu()], dim=1)
        return torch.zeros((len(y), num_classes)).to(self.args.device).scatter_(1, y.reshape(-1, 1), 1.)
    
    def _add_hook(self, module: Module):
        if self.args.model_name == 'vgg16':
            if isinstance(module, MaxPool2d):
                self.hiddens_hooks.append(OutputHook(module))
        elif self.args.model_name in ['ir152', 'facenet64', 'facenet']:
            if isinstance(module, Sequential):
                self.hiddens_hooks.append(OutputHook(module))
        else:
            raise RuntimeError(f'model {self.args.model_name} is not support for BiDO')
        
    def before_train(self):
        super().before_train()
        self.hiddens_hooks.clear()
        traverse_module(self.model, self._add_hook, call_middle=True)
        assert len(self.hiddens_hooks) > 0
        
    def after_train(self):
        super().after_train()
        for hook in self.hiddens_hooks:
            hook.close()
        
        
    def calc_loss(self, inputs: torch.Tensor, result: ModelResult, labels: LongTensor):
        res = result.result
        bs = len(inputs)
        
        total_loss = 0
        cross_loss = F.cross_entropy(res, labels)
        
        total_loss += cross_loss
        
        # hidden_input_losses = []
        # hidden_output_losses = []
        
        h_data = inputs.view(bs, -1)
        h_label = self._to_onehot(labels, NUM_CLASSES[self.args.dataset_name]).to(self.args.device).view(bs, -1)
        
        for hidden_hook in self.hiddens_hooks:
            h_hidden = hidden_hook.get_feature().reshape(bs, -1)
            
            hidden_input_loss, hidden_output_loss = self.objective_fn(h_hidden, h_label, h_data, 5., self.args.kernel_type)
            
            total_loss += self.args.coef_hidden_input * hidden_input_loss
            total_loss += - self.args.coef_hidden_output * hidden_output_loss
        
        return total_loss