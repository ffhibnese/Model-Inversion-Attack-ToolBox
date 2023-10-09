
from dataclasses import dataclass
@dataclass
class MirrorBlackBoxArgs:
    arch_name:str
    genforce_model_name: str
    genforce_checkpoint_dir: str
    target_labels: list
    work_dir: str
    classifiers_checkpoint_dir: str
    batch_size : int
    use_cache : bool
    device: str
    pre_sample_dir: str
    population: int
    
    resolution: int
    
    log_inverval = 10
    mutation_prob = 0.1
    mutation_ce = 0.1
    generation = 100
    p_std_ce = 1
    min_score = 0.95
    
    use_w_space = True
    repeat_w = True