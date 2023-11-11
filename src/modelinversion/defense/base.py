

import torch
import numpy as np
import os
from torch import LongTensor
from ..models import ModelResult, get_model
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from ..utils import DefenseFolderManager, Accumulator
from tqdm import tqdm
from abc import abstractmethod, ABCMeta
from enum import Enum
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# class OptimizerNames(Enum):
#     SGD = "sgd"
#     ADAM = 'adam'
    
class TqdmStrategy(Enum):
    NONE = 'none'
    EPOCH = 'epoch'
    ITER = 'iter'
    

@dataclass
class TrainArgs:
    
    model_name: str
    dataset_name: str
    
    epoch_num: int
    
    defense_type: str = field(default='no_defense', metadata={'help': 'Defense Type, default: no_defense'})
    
    tqdm_strategy: TqdmStrategy = field(default=TqdmStrategy.EPOCH, metadata={'help': 'Where to use tqdm. NONE: no use; EPOCH: the whole training; ITER: in an epoch'})
    
    device: str = field(default='cpu', metadata={'help': 'Device for train. cpu, cuda, cuda:0, ...'})
    


class BaseTrainer(metaclass=ABCMeta):
    
    def __init__(self, args: TrainArgs, folder_manager: DefenseFolderManager, model: Module, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        self.args = args
        
        self.model = model
        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.folder_manager = folder_manager
        
            
        
    @abstractmethod
    def calc_loss(self, result: ModelResult, labels: torch.LongTensor):
        raise NotImplementedError()
    
    def _update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def prepare_input_label(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)
        
        return imgs, labels
        
    def _train_step(self, inputs, labels):
        self.model.train()
        
        result = self.model(inputs)
        
        loss = self.calc_loss(result, labels)
        self._update_step(loss)
        
        return loss
        
    
    def _train_loop(self, dataloader: DataLoader):
            
        loss_accumulator = Accumulator(1)
            
        iter_times = 0
        for i, batch in enumerate(dataloader):
            iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            loss = self._train_step(inputs, labels)
            loss_accumulator.add(loss)
            
        return loss_accumulator.avg(0), iter_times
            
    def train(self, dataloader: DataLoader):
        
        epochs = range(self.args.epoch_num)
        if self.args.tqdm_strategy == TqdmStrategy.EPOCH:
            epochs = tqdm(epochs)
        
        for epoch in epochs:
            if self.args.tqdm_strategy == TqdmStrategy.ITER:
                dataloader = tqdm(dataloader)
            
            self._train_loop(dataloader)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
        # torch.save(self.model.state_dict(), 'aa.pth')
        self.folder_manager.save_target_model_state_dict(self.model, self.args.dataset_name, self.args.model_name, defense_type=self.args.defense_type)

class RegTrainer(BaseTrainer):
    
    def __init__(self, args: TrainArgs, folder_manager: DefenseFolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, criterion = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        
        assert criterion is not None, 'criterion can not be none'
        self.criterion = criterion
        
    def calc_loss(self, result: ModelResult, labels: LongTensor):
        pred_res = result.result
        return self.criterion(pred_res, labels)
    
class VibTrainer(BaseTrainer):
    
    def __init__(self, args: TrainArgs, folder_manager: DefenseFolderManager, model: Module, optimizer: Optimizer, scheduler: LRScheduler=None, criterion = None, beta: float=1e-2, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, scheduler, **kwargs)
        
        assert criterion is not None, 'criterion can not be none'
        self.criterion = criterion
        self.beta = beta
        
    
    def calc_loss(self, result: ModelResult, labels: LongTensor):
        pred_res = result.result
        mu = result.addition_info['mu']
        std = result.addition_info['std']
        cross_loss = self.criterion(pred_res, labels)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + self.beta * info_loss
        return loss
    
