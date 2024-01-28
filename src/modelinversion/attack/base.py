import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
import yaml

from ..models import *
from ..foldermanager import FolderManager
from ..metrics.base import *
from ..utils import DictAccumulator, Accumulator
from ..enums import TqdmStrategy

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

@dataclass
class BaseGANTrainArgs:

    dataset_name: str
    batch_size: int
    epoch_num: int
    
    dis_gen_update_rate: int = 5
    tqdm_strategy: TqdmStrategy = TqdmStrategy.ITER
    defense_type: str = 'no_defense'
    device: str = 'cpu'
    

class BaseGANTrainer(metaclass=ABCMeta):
    
    def __init__(self, args: BaseGANTrainArgs, folder_manager: FolderManager, **kwargs) -> None:
        self.args = args
        self.folder_manager = folder_manager
        
        self.method_name = self.get_method_name()
        self.tag = self.get_tag()
        
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
        
        for iter_time, batch in enumerate(dataloader, start=1):
            # print(f'len batch: {len(batch)}')
            self.before_dis_train_step()
            dis_ret = self.train_dis_step(batch)
            dis_accumulator.add(dis_ret)
            
            if iter_time % self.args.dis_gen_update_rate == 0:
                self.before_gen_train_step()
                gen_ret = self.train_gen_step(batch)
                gen_accumulator.add(gen_ret)
                
        gen_avg = gen_accumulator.avg()
        dis_avg = dis_accumulator.avg()
        # print_context = {
        #     'epoch': epoch,
        #     'generator': gen_avg,
        #     'discriminator': dis_avg
        # }
        print_context = OrderedDict(
            epoch = epoch,
            generator = gen_avg,
            discriminator = dis_avg
        )
        print(yaml.dump(print_context))
        
    def train(self):
        
        self.prepare_training()
        
        trainloader = self.get_trainloader()
        
        epochs = range(self.args.epoch_num)
        if self.args.tqdm_strategy == TqdmStrategy.EPOCH:
            epochs = tqdm(epochs)
            
        for epoch in epochs:
            if self.args.tqdm_strategy == TqdmStrategy.ITER:
                trainloader = tqdm(trainloader)
            
            self._train_loop(trainloader, epoch)
            
        self.save_state_dict()