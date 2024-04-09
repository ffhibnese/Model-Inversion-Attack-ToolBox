from dataclasses import dataclass

from ..base import BaseAttackConfig


@dataclass
class MirrorBlackboxAttackConfig(BaseAttackConfig):
    genforce_name: str = 'stylegan_celeba_partial256'
    presample_batch_size: int = 64
    population: int = 1000

    # fix params

    log_inverval = 10
    mutation_prob = 0.1
    mutation_ce = 0.1
    generation = 100
    p_std_ce = 1
    min_score = 0.95

    use_w_space = True
    repeat_w = True


@dataclass
class MirrorWhiteboxAttackConfig(BaseAttackConfig):
    genforce_name: str = 'stylegan_celeba_partial256'
    presample_batch_size: int = 64

    gen_num_per_target: int = 5
    do_flip: bool = False
    # use_cache : bool
    loss_class_ce: float = 1
    epoch_num: int = 5000
    lr: float = 0.2
    save_every: int = 100
    use_dropout: bool = False
    latent_space: str = 'w'
    p_std_ce: int = 1
    z_std_ce: int = 1


# @dataclass
# class MirrorBlackBoxConfig:

#     target_name: str
#     eval_name: str
#     genforce_name: str

#     ckpt_dir: str
#     cache_dir: str
#     result_dir: str
#     dataset_dir: str

#     dataset_name: str

#     target_labels: list
#     batch_size: int
#     device: str


# @dataclass
# class MirrorWhiteBoxConfig:

#     target_name: str
#     eval_name: str
#     genforce_name: str

#     ckpt_dir: str
#     cache_dir: str
#     result_dir: str
#     dataset_dir: str

#     dataset_name: str

#     target_labels: list
#     batch_size: int
#     device: str
