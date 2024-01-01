from dataclasses import dataclass
from ..base import BaseAttackConfig

@dataclass
class RLBMIAttackConfig(BaseAttackConfig):
    gan_dataset_name: str = 'celeba'
    # gen_num_per_target: int = 5
    
    z_dim = 100
    alpha = 0
    iter_times = 40000
    max_step = 1