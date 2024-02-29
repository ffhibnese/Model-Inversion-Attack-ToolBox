from dataclasses import dataclass, field

from ..base import BaseAttackConfig

@dataclass
class KEDMIAttackConfig(BaseAttackConfig):
    
    gan_dataset_name: str = 'celeba'
    gan_target_name: str = 'vgg16'
    gen_num_per_target: int = 5
    
    lr: float = 2e-2
    iter_times: int = 1500
    
    # fixed params
    coef_iden_loss=100
    
    clip_range=1
    z_dim = 150