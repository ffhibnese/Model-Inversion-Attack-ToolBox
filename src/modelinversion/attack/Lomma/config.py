from dataclasses import dataclass, field

from ..GMI.config import GMIAttackConfig

@dataclass
class LommaGMIAttackConfig(GMIAttackConfig):
    
    aug_model_dataset_name: str = 'celeba'
    aug_model_names: list = field(default_factory = lambda : ['efficientnet_b0','efficientnet_b1', 'efficientnet_b2'])
    preg_generate_batch_size: int = 64