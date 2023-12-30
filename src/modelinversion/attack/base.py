import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import torch

from ..models import *
from ..foldermanager import FolderManager
from ..metrics.base import *
from ..utils import DictAccumulator

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
        
        # os.makedirs(cache_dir)
        self.folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, cache_dir, result_dir, config.defense_ckpt_dir, config.defense_type)
        
        self.prepare_classifiers()
        
    def register_dirs(self, dirs: dict):
        # self.folder_manager.config.
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
        
   
    
    def attack(self, batch_size: int, target_labels: list):
        
        
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