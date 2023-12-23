import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from ..utils import FolderManager

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
    batch_size: int = 20
    target_labels: list = field(default_factory = lambda : list(range(300)))
    defense_type: str = 'no_defense'
    device: str = 'cpu'
    
@dataclass
class BaseAttackArgs:
    taregt_name: str
    eval_name: str
    target_labels: list
    device: str
    

class BaseAttacker(metaclass=ABCMeta):
    
    def __init__(self, config: BaseAttackConfig) -> None:
        self.config = config
        self.folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, config.result_dir, )
        
    def parse_config(self, config: BaseAttackConfig) -> BaseAttackArgs:
        return BaseAttackArgs(config.target_name, config.eval_name)