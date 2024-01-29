import os
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch import nn, LongTensor
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from .models import *
from .foldermanager import FolderManager
from .metrics.base import *
from .utils import DictAccumulator, Accumulator, print_as_yaml
from .enums import TqdmStrategy

@dataclass
class BaseGANTrainArgs:

    dataset_name: str
    batch_size: int
    epoch_num: int
    
    dis_gen_update_rate: int = 5
    tqdm_strategy: TqdmStrategy = TqdmStrategy.ITER
    defense_type: str = 'no_defense'
    device: str = 'cpu'

@dataclass
class BaseTrainArgs:
    
    model_name: str
    dataset_name: str
    
    epoch_num: int
    
    clip_grad_norm: Optional[float] = None
    
    defense_type: str = field(default='no_defense', metadata={'help': 'Defense Type, default: no_defense'})
    
    tqdm_strategy: TqdmStrategy = field(default=TqdmStrategy.ITER, metadata={'help': 'Where to use tqdm. NONE: no use; EPOCH: the whole training; ITER: in an epoch'})
    
    device: str = field(default='cpu', metadata={'help': 'Device for train. cpu, cuda, cuda:0, ...'})
    

class BaseTrainer(metaclass=ABCMeta):
    
    def __init__(self, args: BaseTrainArgs, folder_manager: FolderManager, model: BaseTargetModel, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        self.args = args
        
        self.model = model
        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.folder_manager = folder_manager
        
        self._epoch = 0
        self._iteration = 0
        
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def iteration(self):
        return self._iteration    
            
        
    @abstractmethod
    def calc_loss(self, inputs, result: ModelResult, labels: torch.LongTensor):
        raise NotImplementedError()
    
    def calc_acc(self, inputs, result: ModelResult, labels: torch.LongTensor):
        res = result.result
        assert res.ndim <= 2
        
        pred = torch.argmax(res, dim=-1)
        # print((pred == labels).float())
        return (pred == labels).float().mean()
    
    def _update_step(self, loss):
        self.optimizer.zero_grad()
        if self.args.clip_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_norm)
        loss.backward()
        self.optimizer.step()
        
    def prepare_input_label(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)
        
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
        for i, batch in enumerate(dataloader):
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
            
        for i, batch in enumerate(dataloader):
            self._iteration = i
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._test_step(inputs, labels)
            accumulator.add(step_res)
            
        return accumulator.avg()
            
    def train(self, trainloader: DataLoader, testloader: DataLoader = None):
        
        epochs = range(self.args.epoch_num)
        if self.args.tqdm_strategy == TqdmStrategy.EPOCH:
            epochs = tqdm(epochs)
        
        for epoch in epochs:
            if self.args.tqdm_strategy == TqdmStrategy.ITER:
                trainloader = tqdm(trainloader)
            
            self._epoch = epoch
            
            train_res = self._train_loop(trainloader)
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})
            
            if testloader is not None:
                if self.args.tqdm_strategy == TqdmStrategy.ITER:
                    testloader = tqdm(testloader)
                test_res = self._test_loop(testloader)
                print_as_yaml({'test': test_res})
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            if epoch % 10 == 9:
                self.save_state_dict()
                
        self.save_state_dict()


    def save_state_dict(self):
        model = self.model
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        self.folder_manager.save_target_model_state_dict(model, self.args.dataset_name, self.args.model_name)



@dataclass
class BaseAttackConfig:
    
    # classifier
    target_name: str
    eval_name: str
    
    # folders
    ckpt_dir: str
    result_dir: str
    dataset_dir: str
    cache_dir: str
    defense_ckpt_dir: str = None
    
    # dataset
    dataset_name: str = 'celeba'
    
    # misc
    defense_type: str = 'no_defense'
    device: str = 'cpu'

class BaseAttacker(metaclass=ABCMeta):
    
    def __init__(self, config: BaseAttackConfig) -> None:
        self.config = config
        tag = self.get_tag()
        cache_dir = os.path.join(config.cache_dir, tag)
        result_dir = os.path.join(config.result_dir, tag)
        
        self.folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, cache_dir, result_dir, config.defense_ckpt_dir, config.defense_type)
        
        self.prepare_classifiers()
        
        print('--------------- config --------------')
        print(config)
        print('-------------------------------------')
        
    def register_dirs(self, dirs: dict):
        for k, v in dirs.items():
            os.makedirs(v, exist_ok=True)
            setattr(self.folder_manager.config, k, v)
    
    @abstractmethod
    def get_tag(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def prepare_attack(self):
        raise NotImplementedError()
    
    @abstractmethod
    def attack_step(self, iden) -> dict:
        raise NotImplementedError()
    
    def prepare_classifiers(self):
        
        config = self.config
        folder_manager = self.folder_manager
        
        self.T = get_model(config.target_name, config.dataset_name, device=config.device, defense_type=config.defense_type)
        folder_manager.load_target_model_state_dict(self.T, config.dataset_name, config.target_name, device=config.device, defense_type=config.defense_type)

        self.E = get_model(config.eval_name, config.dataset_name, device=config.device)
        folder_manager.load_target_model_state_dict(self.E, config.dataset_name, config.eval_name, device=config.device)
        
        self.T.eval()
        self.E.eval()
   
    
    def attack(self, batch_size: int, target_labels: list):
        
        self.batch_size = batch_size
        self.target_labels = target_labels
        
        self.prepare_attack()
        
        config = self.config
        
        print("=> Begin attacking ...")
        # aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
        
        total_num = len(target_labels)
        
        accumulator = DictAccumulator()
        
        if total_num > 0:
            for idx in range((total_num - 1) // batch_size + 1):
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                iden = torch.tensor(
                    target_labels[idx * batch_size: min((idx+1)*batch_size, total_num)], device=config.device, dtype=torch.long
                )
                
                update_dict = self.attack_step(iden)
                
                bs = len(iden)
                
                accumulator.add(update_dict)
                
            for key, val in accumulator.avg().items():
                print(f'average {key}: {val:.6f}')
            
    def evaluation(self, batch_size, transform=None, knn=True, feature_distance=True, fid=False):
        eval_metrics = []
        
        if knn:
            eval_metrics.append(KnnDistanceMetric(self.folder_manager, device=self.config.device, model=self.E))
        
        if feature_distance:
            eval_metrics.append(FeatureDistanceMetric(self.folder_manager, self.config.device, model=self.E))
        
        if fid:
            eval_metrics.append(FIDMetric(self.folder_manager, device=self.config.device, model=None))
                                
        for metric in eval_metrics:
            metric: BaseMetric
            print(f'calculate {metric.get_metric_name()}')
            metric.evaluation(self.config.dataset_name, batch_size, transform)
            
class BaseSingleLabelAttacker(BaseAttacker):

    def __init__(self, config: BaseAttackConfig) -> None:
        super().__init__(config)
        
    def attack_step(self, target) -> dict:
        return super().attack_step(target)
        
    def attack(self, batch_size: int, target_labels: list):
    
        self.batch_size = batch_size
        self.target_labels = target_labels
        
        self.prepare_attack()
        
        config = self.config
        
        print("=> Begin attacking ...")
        
        total_num = len(target_labels)
        
        accumulator = DictAccumulator()
        
        if total_num > 0:
            for target in target_labels:
                print(f"--------------------- Attack label [{target}]------------------------------")
                
                update_dict = self.attack_step(target)
                
                
                accumulator.add(update_dict)
                
            for key, val in accumulator.avg().items():
                print(f'average {key}: {val:.6f}')


    

class BaseGANTrainer(metaclass=ABCMeta):
    
    def __init__(self, args: BaseGANTrainArgs, folder_manager: FolderManager, **kwargs) -> None:
        self.args = args
        self.folder_manager = folder_manager
        
        self.method_name = self.get_method_name()
        self.tag = self.get_tag()
        self._epoch = 0
        self._iteration = 0
        
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def iteration(self):
        return self._iteration
        
    @abstractmethod
    def get_method_name(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_tag(self) -> str:
        raise NotImplementedError()
        
    @abstractmethod
    def prepare_training(self):
        # raise NotImplementedError()
        self.G = None
        self.D = None
        
    @abstractmethod
    def get_trainloader(self) -> DataLoader:
        raise NotImplementedError()
        
    @abstractmethod
    def train_gen_step(self, batch) -> OrderedDict:
        raise NotImplementedError()
    
    @abstractmethod
    def train_dis_step(self, batch) -> OrderedDict:
        raise NotImplementedError()

    def before_train(self):
        pass
    
    def after_train(self):
        pass
    
    def before_gen_train_step(self):
        # self.model.train()
        self.G.train()
        # self.D.eval()
        
    def before_dis_train_step(self):
        # self.G.eval()
        self.D.train()
        
    
    
    def save_state_dict(self):
        self.folder_manager.save_state_dict(self.G, [self.method_name, f'{self.tag}_G.pt'], self.args.defense_type)
        self.folder_manager.save_state_dict(self.D, [self.method_name, f'{self.tag}_D.pt'], self.args.defense_type)
    
    def loss_update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    def _train_loop(self, dataloader: DataLoader, epoch):
        self.before_train()
        
        gen_accumulator = DictAccumulator()
        dis_accumulator = DictAccumulator()
        
        for iter_time, batch in enumerate(dataloader):
            self._iteration += 1
            self.before_dis_train_step()
            dis_ret = self.train_dis_step(batch)
            dis_accumulator.add(dis_ret)
            
            if (iter_time + 1) % self.args.dis_gen_update_rate == 0:
                self.before_gen_train_step()
                gen_ret = self.train_gen_step(batch)
                gen_accumulator.add(gen_ret)
                
        gen_avg = gen_accumulator.avg()
        dis_avg = dis_accumulator.avg()
        
        print_context = OrderedDict(
            epoch = epoch,
            generator = gen_avg,
            discriminator = dis_avg
        )
        # print(yaml.dump(print_context))
        print_as_yaml(print_context)
        
    def train(self):
        
        self.prepare_training()
        
        trainloader = self.get_trainloader()
        
        self._epoch = 0
        self._iteration = 0
        
        epochs = range(self.args.epoch_num)
        if self.args.tqdm_strategy == TqdmStrategy.EPOCH:
            epochs = tqdm(epochs)
            
        for epoch in epochs:
            self._epoch = epoch
            if self.args.tqdm_strategy == TqdmStrategy.ITER:
                trainloader = tqdm(trainloader)
            
            self._train_loop(trainloader, epoch)
            
        self.save_state_dict()