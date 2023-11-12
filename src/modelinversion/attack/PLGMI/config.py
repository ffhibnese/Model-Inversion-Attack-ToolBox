from dataclasses import dataclass, field

@dataclass
class PlgmiAttackConfig:
    
    target_name: str
    cgan_target_name: str
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

    # default parameter
    batch_size: int = 20
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    # gen_num_features: int = 64
    # gen_dim_z: int = 128
    # gen_bottom_width: int = 4
    gen_distribution: str = 'normal'