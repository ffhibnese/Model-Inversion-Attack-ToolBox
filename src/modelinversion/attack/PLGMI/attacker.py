import os
from dataclasses import dataclass, field

import torch

from ..base import BaseAttacker, BaseAttackConfig
from .code.reconstruct import inversion, PlgmiArgs
from .config import PlgmiAttackConfig
from ...utils import set_random_seed
from ...foldermanager import FolderManager
from .code.models.generators.resnet64 import ResNetGenerator
from ...models import *

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
    
    # fixed params
    gen_num_features = 64
    gen_dim_z = 128
    gen_bottom_width = 4

class PLGMIAttacker(BaseAttacker):
    
    def __init__(self, config: PLGMIAttackConfig) -> None:
        super().__init__(config)
        self.config: PLGMIAttackConfig
        
    def prepare_attack_models(self):
        config: PLGMIAttackConfig = self.config
        self.G = ResNetGenerator(num_classes=NUM_CLASSES[config.dataset_name], distribution=config.gen_distribution)
        
        self.folder_manager.load_state_dict(
            self.G, 
            ['PLGMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_PLG_MI_G.tar'], 
            device=config.device
        )
        
    def attack_step(self, iden):
        return inversion(
            self.config, self.G, self.T, self.E, iden, self.folder_manager, 
            self.config.lr, self.config.iter_times, 5
        )
        