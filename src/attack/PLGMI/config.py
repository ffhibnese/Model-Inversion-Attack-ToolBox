from dataclasses import dataclass

@dataclass
class PlgmiAttackConfig:
    
    target_name: str
    cgan_target_name: str
    eval_name: str
    ckpt_dir: str
    result_dir: str
    
    dataset_name: str
    
    device: str

    # default parameter
    batch_size: int = 20
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_num_features: int = 64
    gen_dim_z: int = 128
    gen_bottom_width: int = 4
    gen_distribution: str = 'normal'