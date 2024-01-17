from dataclasses import dataclass, field
from ..base import BaseAttackConfig

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
