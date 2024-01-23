from dataclasses import dataclass, field

from ..base import BaseAttackConfig

# @dataclass
# class PPAAttackConfig(BaseAttackConfig):
    
#     stylegan_resp_dir: str = '.'
#     stylegan_file_path: str = 'ffhq.pkl'
    
#     initial_sample_num: int = 5000
#     candidate_num: int = 200
#     gen_num_per_target: int = 50
#     final_selection_iters: int = 100
    
#     seed: int = 42
#     truncation_psi: float = 0.5
#     truncation_cutoff: int= 8