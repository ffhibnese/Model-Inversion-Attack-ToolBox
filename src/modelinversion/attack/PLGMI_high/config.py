from dataclasses import dataclass, field
from typing import Callable

from kornia import augmentation
from ..base import BaseAttackConfig

@dataclass
class PLGMIAttackConfig(BaseAttackConfig):
    
    # gan config
    gan_target_name: str = 'vgg16'
    gan_dataset_name: str = 'celeba'
    
    # attack config
    inv_loss_type: str = 'max_margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_distribution: str = 'normal'
    gen_num_per_target: int = 5
    
    attack_transform: Callable = augmentation.container.ImageSequential(
            augmentation.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            augmentation.ColorJitter(brightness=0.2, contrast=0.2),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(5),
        )
    
    # fixed params
    gen_num_features = 64
    gen_dim_z = 128
    gen_bottom_width = 4
