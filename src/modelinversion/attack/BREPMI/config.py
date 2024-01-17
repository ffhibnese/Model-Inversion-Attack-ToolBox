from dataclasses import dataclass, field

from ..base import BaseAttackConfig

@dataclass
class BrepAttackConfig(BaseAttackConfig):
    
    gan_target_name: str = 'vgg16'
    gan_dataset_name: str = 'celeba'
    # z_init_batch_size: str = 64
    
    sphere_points_count : int = 32
    init_sphere_radius : float = 2
    sphere_expansion_coeff : float = 1.3
    point_clamp_min : float = -1.5
    point_clamp_max : float = 1.5
    max_iters_at_radius_before_terminate : int = 1000
    
    batch_dim_for_initial_points : int = 256
    repulsion_only: bool = True
    init_z_max_iter: int = 10000
    
    z_dim = 100

# @dataclass
# class BrepAttackConfig:
    
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