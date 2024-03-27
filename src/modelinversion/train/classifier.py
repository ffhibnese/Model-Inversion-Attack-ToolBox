import os
import importlib
from dataclasses import field, dataclass
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Optional, Iterator, Tuple, Callable, Sequence
import math

import torch
from torch import nn, Tensor, LongTensor
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from ..models import BaseImageClassifier
from ..utils import unwrapped_parallel_module, ClassificationLoss, obj_to_yaml, print_as_yaml, DictAccumulator


@dataclass
class BaseTrainConfig:
    
    experiment_dir: str
    save_name: str
    device: torch.device
    
    model: BaseImageClassifier
    optimizer: Optimizer
    lr_scheduler: Optional[LRScheduler] = None
    clip_grad_norm: Optional[float] = None
    
    save_per_epochs: int = 10
    

class BaseTrainer(ABC):
    
    def __init__(self, config: BaseTrainConfig, *args, **kwargs) -> None:
        self.config = config
        os.makedirs(config.experiment_dir, exist_ok=True)
        self.save_path = os.path.join(config.experiment_dir, config.save_name)
        
        self._epoch = 0
        self._iteration = 0
        
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def iteration(self):
        return self._iteration    
    
    @property
    def model(self):
        return self.config.model
    
    @property
    def optimizer(self):
        return self.config.optimizer
    
    @property
    def lr_scheduler(self):
        return self.config.lr_scheduler
            
        
    @abstractmethod
    def calc_loss(self, inputs, result, labels: torch.LongTensor):
        raise NotImplementedError()
    
    def calc_acc(self, inputs, result, labels: torch.LongTensor):
        res = result[0]
        assert res.ndim <= 2
        
        pred = torch.argmax(res, dim=-1)
        # print((pred == labels).float())
        return (pred == labels).float().mean()
    
    def _update_step(self, loss):
        self.optimizer.zero_grad()
        if self.config.clip_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
        loss.backward()
        self.optimizer.step()
        
    def prepare_input_label(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.config.device)
        labels = labels.to(self.config.device)
        
        return imgs, labels
        
    def _train_step(self, inputs, labels) -> OrderedDict:

        self.before_train_step()
        
        result = self.model(inputs)
        
        loss = self.calc_loss(inputs, result, labels)
        acc = self.calc_acc(inputs, result, labels)
        self._update_step(loss)
        
        return OrderedDict(
            loss = loss,
            acc = acc
        )
        
    def before_train(self):
        pass
    
    def after_train(self):
        pass
    
    def before_train_step(self):
        self.model.train()
        
    def before_test_step(self):
        self.model.eval()
        
    
    def _train_loop(self, dataloader: DataLoader):
        
        self.before_train()
            
        accumulator = DictAccumulator()
            
        # iter_times = 0
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            # iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._train_step(inputs, labels)
            accumulator.add(step_res)
            
        self.after_train()
            
        return accumulator.avg()
    
    @torch.no_grad()
    def _test_step(self, inputs, labels):
        # self.model.eval()
        self.before_test_step()
        
        result = self.model(inputs)
        
        acc = self.calc_acc(inputs, result, labels)
        
        return OrderedDict(acc = acc)
    
    @torch.no_grad()
    def _test_loop(self, dataloader: DataLoader):
        
        
        accumulator = DictAccumulator()
            
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._test_step(inputs, labels)
            accumulator.add(step_res)
            
        return accumulator.avg()
            
    def train(self, epoch_num: int, trainloader: DataLoader, testloader: DataLoader = None):
        
        epochs = range(epoch_num)
        
        for epoch in epochs:
            
            self._epoch = epoch
            
            train_res = self._train_loop(trainloader)
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})
            
            if testloader is not None:
                test_res = self._test_loop(testloader)
                print_as_yaml({'test': test_res})
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            if (epoch+1) % self.config.save_per_epochs == 0:
                self.save_state_dict()
                
        self.save_state_dict()


    def save_state_dict(self):
        model = self.model
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            model = self.model.module
            
        torch.save({'state_dict': model.state_dict()}, self.save_path)

@dataclass
class SimpleTrainConfig(BaseTrainConfig):
    
    loss_fn: str | Callable = 'cross_entropy'

class SimpleTrainer(BaseTrainer):
    
    def __init__(self, config: BaseTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        
        self.loss_fn = ClassificationLoss(config.loss_fn)
    
    def calc_loss(self, inputs, result, labels: LongTensor):
        return self.loss_fn(result[0], labels)
    
    
    
    
    
    
    
    