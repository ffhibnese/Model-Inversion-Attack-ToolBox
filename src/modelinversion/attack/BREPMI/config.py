from dataclasses import dataclass, field

@dataclass
class BrepAttackConfig:
    
    target_name: str
    gan_target_name: str
    eval_name: str
    ckpt_dir: str
    result_dir: str
    dataset_dir: str
    cache_dir: str
    
    dataset_name: str
    gan_dataset_name: str
    
    batch_size: int = 60
    target_labels: list = field(default_factory = lambda : list(range(300)))
    device: str = 'cpu'
    
    defense_type: str = 'no_defense'
    defense_ckpt_dir: str= None