from dataclasses import dataclass, field

from ..GMI.config import GMIAttackConfig
from ..KEDMI.config import KEDMIAttackConfig

@dataclass
class LommaGMIAttackConfig(GMIAttackConfig):
    
    aug_model_dataset_name: str = 'celeba'
    aug_model_names: list = field(default_factory = lambda : ['efficientnet_b0','efficientnet_b1', 'efficientnet_b2'])
    preg_generate_batch_size: int = 64
    
@dataclass
class LommaKEDMIAttackConfig(KEDMIAttackConfig):
    aug_model_dataset_name: str = 'celeba'
    aug_model_names: list = field(default_factory = lambda : ['efficientnet_b0','efficientnet_b1', 'efficientnet_b2'])
    preg_generate_batch_size: int = 64
    
    reg_coef = 1