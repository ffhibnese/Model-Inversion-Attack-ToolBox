from dataclasses import dataclass

@dataclass
class GmiAttackConfig:
    
    target_name: str
    eval_name: str
    gan_target_name: str
    ckpt_dir: str
    result_dir: str
    
    dataset_name: str
    
    batch_size: int
    device: str