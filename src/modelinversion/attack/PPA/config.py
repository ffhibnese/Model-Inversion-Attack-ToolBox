from dataclasses import dataclass, field

from ..base import BaseAttackConfig

@dataclass
class PPAAttackConfig(BaseAttackConfig):
    
    stylegan_resp_dir: str = '.'
    stylegan_file_path: str = 'ffhq.pkl'
    candidate_num: int = 200
    
    seed: int = 42
    truncation_psi: float=0.7
    truncation_cutoff: int=18