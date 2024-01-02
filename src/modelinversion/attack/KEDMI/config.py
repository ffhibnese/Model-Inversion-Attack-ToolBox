from dataclasses import dataclass, field

from ..base import BaseAttackConfig

@dataclass
class KEDMIAttackConfig(BaseAttackConfig):
    
    gan_dataset_name: str = 'celeba'
    gan_target_name: str = 'vgg16'
    gen_num_per_target: int = 5
    
    # fixed params
    
    lr=2e-2
    # momentum=0.9
    lamda=100
    iter_times=1500
    clip_range=1
    # num_seeds=5
    z_dim = 100