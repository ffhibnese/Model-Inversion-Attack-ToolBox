

import torch
import numpy as np
import os
from torch import LongTensor
from ..models import ModelResult, get_model
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from ..utils import FolderManager, Accumulator
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
class BaseTrainArgs:
    
    model_name: str
    dataset_name: str
    
    epoch_num: int
    
    defense_type: str = field(default='no_defense', metadata={'help': 'Defense Type, default: no_defense'})
    
    tqdm_strategy: TqdmStrategy = field(default=TqdmStrategy.ITER, metadata={'help': 'Where to use tqdm. NONE: no use; EPOCH: the whole training; ITER: in an epoch'})
    
    device: str = field(default='cpu', metadata={'help': 'Device for train. cpu, cuda, cuda:0, ...'})
    


class BaseTrainer(metaclass=ABCMeta):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: DefenseFolderManager, model: Module, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        self.args = args
        
        self.model = model
        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.folder_manager = folder_manager
        
            
        
    @abstractmethod
    def calc_loss(self, inputs, result: ModelResult, labels: torch.LongTensor):
        raise NotImplementedError()
    
    def calc_acc(self, result: ModelResult, labels: torch.LongTensor):
        res = result.result
        assert res.ndim <= 2
        
        pred = torch.argmax(res, dim=-1)
        # print((pred == labels).float())
        return (pred == labels).float().mean()
    
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
        
        loss = self.calc_loss(inputs, result, labels)
        acc = self.calc_acc(result, labels)
        self._update_step(loss)
        
        return {
            'loss': loss,
            'acc': acc
        }
        
    
    def _train_loop(self, dataloader: DataLoader):
            
        self.model.train()
        accumulator = Accumulator(2)
            
        iter_times = 0
        for i, batch in enumerate(dataloader):
            iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._train_step(inputs, labels)
            loss = step_res['loss']
            acc = step_res['acc']
            accumulator.add(loss, acc)
            
        return accumulator.avg()
    
    @torch.no_grad()
    def _test_step(self, inputs, labels):
        self.model.eval()
        
        result = self.model(inputs)
        
        acc = self.calc_acc(result, labels)
        
        return {
            'acc': acc
        }
    
    @torch.no_grad()
    def _test_loop(self, dataloader: DataLoader):
        
        
        accumulator = Accumulator(1)
            
        iter_times = 0
        for i, batch in enumerate(dataloader):
            iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._test_step(inputs, labels)
            acc = step_res['acc']
            accumulator.add(acc)
            
        return accumulator.avg()
            
    def train(self, trainloader: DataLoader, testloader: DataLoader = None):
        
        epochs = range(self.args.epoch_num)
        if self.args.tqdm_strategy == TqdmStrategy.EPOCH:
            epochs = tqdm(epochs)
        
        for epoch in epochs:
            if self.args.tqdm_strategy == TqdmStrategy.ITER:
                trainloader = tqdm(trainloader)
                    
            
            loss, acc = self._train_loop(trainloader)
            print(f'epoch {epoch}\t train loss: {loss:.6f}\t train acc: {acc:.6f}')
            if testloader is not None:
                if self.args.tqdm_strategy == TqdmStrategy.ITER:
                    testloader = tqdm(testloader)
                test_acc, = self._test_loop(testloader)
                print(f'epoch {epoch}\t test acc: {test_acc:.6f}')
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
        self.folder_manager.save_target_model_state_dict(self.model, self.args.dataset_name, self.args.model_name)


    

