from dataclasses import dataclass


@dataclass
class DeepInversionConfig:

    target_name: str
    eval_name: str
    target_labels: list

    cache_dir: str
    result_dir: str
    dataset_name: str

    device: str

    batch_size: int = 60
    r_feature: float = 0.01
    do_flip: bool = True
    adi_scale: float = 0.0
    lr: float = 0.25
