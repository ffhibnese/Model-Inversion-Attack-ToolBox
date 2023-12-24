from dataclasses import dataclass, field

import torch

from ..base import BaseAttacker, BaseAttackConfig, BaseAttackArgs

@dataclass
class PLGMIAttackConfig(BaseAttackConfig):
    
    # gan config
    gan_target_name: str = 'vgg16'
    gan_dataset_name: str = 'celeba'
    
    # attack config
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_distribution: str = 'normal'
    
@dataclass
class PLGMIAttackArgs(BaseAttackArgs):
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_num_features: int = 64
    gen_dim_z: int = 128
    gen_bottom_width: int = 4
    gen_distribution: str = 'normal'


class PLGMIAttacker(BaseAttacker):
    
    def __init__(self, config: BaseAttackConfig) -> None:
        super().__init__(config)
        
    def parse_config(self, config: BaseAttackConfig) -> BaseAttackArgs:
        # return super().parse_config(config)
        pass
        
    # def prepare_attack_models(self):
        
        