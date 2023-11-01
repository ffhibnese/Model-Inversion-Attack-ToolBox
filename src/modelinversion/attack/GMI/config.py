from dataclasses import dataclass, field

@dataclass
class GmiAttackConfig:
    
    target_name: str
    eval_name: str
    gan_target_name: str
    ckpt_dir: str
    result_dir: str
    
    dataset_name: str
    gan_dataset_name: str
    
    batch_size: int = 60
    target_labels: list = field(default_factory = lambda : list(range(300)))
    device: str = 'cpu'