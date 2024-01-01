from dataclasses import dataclass, field

from ..base import BaseAttackConfig

@dataclass
class GMIAttackConfig(BaseAttackConfig):
    
    gan_dataset_name: str = 'celeba'
    gen_num_per_target: int = 5
    
    # fixed params
    
    lr=2e-2
    momentum=0.9
    lamda=100
    iter_times=1500
    clip_range=1
    num_seeds=5
    
    

# @dataclass
# class GmiAttackConfig:
    
#     target_name: str
#     eval_name: str
#     # gan_target_name: str
#     ckpt_dir: str
#     dataset_dir: str
#     result_dir: str
#     cache_dir: str
    
#     dataset_name: str
#     gan_dataset_name: str
    
#     batch_size: int = 60
#     target_labels: list = field(default_factory = lambda : list(range(300)))
#     device: str = 'cpu'
    
#     defense_type: str = 'no_defense'
#     defense_ckpt_dir: str= None