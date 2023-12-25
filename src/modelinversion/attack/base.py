import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import torch

from ..models import *
from ..foldermanager import FolderManager

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
    
@dataclass
class BaseAttackArgs:
    taregt_name: str
    eval_name: str
    device: str
    

class BaseAttacker(metaclass=ABCMeta):
    
    def __init__(self, config: BaseAttackConfig) -> None:
        self.config = config
        # self.args = self.parse_config(config)
        self.folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, config.result_dir, config.defense_ckpt_dir, config.defense_type)
    
    def prepare_classifiers(self):
        
        config = self.config
        folder_manager = self.folder_manager
        
        self.T = get_model(config.target_name, config.dataset_name, device=config.device, defense_type=config.defense_type)
        folder_manager.load_target_model_state_dict(self.T, config.dataset_name, config.target_name, device=config.device, defense_type=config.defense_type)

        self.E = get_model(config.eval_name, config.dataset_name, device=config.device)
        folder_manager.load_target_model_state_dict(self.E, config.dataset_name, config.eval_name, device=config.device)
        
    @abstractmethod
    def prepare_attack_models(self):
        raise NotImplementedError()
    
    @abstractmethod
    def attack_step(self, iden):
        raise NotImplementedError()
    
    def attack(self, batch_size: int, target_labels: list):
        
        config = self.config
        
        print("=> Begin attacking ...")
        aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
        
        total_num = len(target_labels)
        
        if total_num > 0:
            for idx in range((total_num - 1) // batch_size + 1):
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                iden = torch.tensor(
                    target_labels[idx * batch_size: min((idx+1)*batch_size, total_num)], device=config.device, dtype=torch.long
                )
                
                acc, acc5, var, var5 = self.attack_step(iden)
                
                bs = len(iden)
                
                aver_acc += acc * bs / total_num
                aver_acc5 += acc5  * bs / total_num
                aver_var += var  * bs / total_num
                aver_var5 += var5  * bs / total_num
                
            print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,aver_acc5,aver_var,aver_var5))
            
    