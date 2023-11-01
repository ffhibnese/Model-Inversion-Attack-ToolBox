from dataclasses import dataclass

@dataclass()
class MirrorBlackBoxConfig:
    
    target_name: str
    eval_name: str
    genforce_name: str
    
    ckpt_dir: str
    cache_dir: str
    result_dir: str
    dataset_dir: str
    
    dataset_name: str
    
    target_labels: list
    batch_size: int
    device: str
    
    
@dataclass
class MirrorWhiteBoxConfig:
    
    target_name: str
    eval_name: str
    genforce_name: str
    
    ckpt_dir: str
    cache_dir: str
    result_dir: str
    dataset_dir: str
    
    dataset_name: str
    
    target_labels: list
    batch_size: int
    device: str