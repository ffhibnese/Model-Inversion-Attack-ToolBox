from abc import ABC, abstractmethod
from typing import Sequence

import torch

class BaseLatentsSampler(ABC):
    """The base class for latent vectors samplers.
    """
    
    @abstractmethod
    def __call__(self, sample_num: int, batch_size: int):
        """The sampling function of the sampler.

        Args:
            sample_num (int): The number of latent vectors sampled.
            batch_size (int): Batch size for sampling.
        """
        pass
    
class SimpleLatentsSampler(BaseLatentsSampler):
    """A latent vector sampler that generates Gaussian distributed random latent vectors with the given `input_size`.

    Args:
        input_size (int or Sequence[int]): The shape of the latent vectors.
    """
    
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