from abc import ABC, abstractmethod
from typing import Sequence

import torch

class BaseLatentsSampler(ABC):
    
    @abstractmethod
    def __call__(self, sample_num: int, batch_size: int):
        pass
    
class SimpleLatentsSampler(BaseLatentsSampler):
    
    def __init__(self, input_size: int | Sequence[int]) -> None:
        super().__init__()
        
        if isinstance(input_size, int):
            input_size = (input_size, )
        if not isinstance(input_size, tuple):
            input_size = tuple(input_size)
        self._input_size = input_size
        
    def __call__(self, sample_num: int, batch_size: int):
       size = (sample_num, ) + self._input_size
       return torch.randn(size)