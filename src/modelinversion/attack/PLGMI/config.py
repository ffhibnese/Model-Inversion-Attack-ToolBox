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

# @dataclass
# class PlgmiAttackConfig:
    
#     target_name: str
#     gan_target_name: str
#     eval_name: str
#     ckpt_dir: str
#     result_dir: str
#     dataset_dir: str
#     cache_dir: str
    
#     dataset_name: str
#     gan_dataset_name: str
    
#     batch_size: int = 60
#     target_labels: list = field(default_factory = lambda : list(range(300)))
#     device: str = 'cpu'
    
#     defense_type: str = 'no_defense'
#     defense_ckpt_dir: str= None

#     # default parameter
#     batch_size: int = 20
#     inv_loss_type: str = 'margin'
#     lr: float = 0.1
#     iter_times: int = 600
#     # gen_num_features: int = 64
#     # gen_dim_z: int = 128
#     # gen_bottom_width: int = 4
#     gen_distribution: str = 'normal'