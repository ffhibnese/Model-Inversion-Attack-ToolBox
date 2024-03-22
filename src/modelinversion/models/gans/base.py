from abc import abstractmethod
from typing import Union, Optional

import torch
import torch.nn as nn

class LambdaModule(nn.Module):
    
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)

class BaseImageGenerator(nn.Module):
    
    def __init__(self, resolution, input_size: Union[int, tuple[int]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._resolution = resolution
        if isinstance(input_size, int):
            input_size = (input_size, )
        self._input_size = input_size
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def input_size(self):
        return self._input_size
    
    @abstractmethod
    def sample_latents(self, sample_num: int, batch_size: int):
        pass
    
    @abstractmethod
    def forward(self, *inputs, labels=None, **kwargs):
        pass
    
class BaseIntermediateImageGenerator(BaseImageGenerator):
    
    def __init__(self, resolution, input_size: int | tuple[int], block_num, *args, **kwargs) -> None:
        super().__init__(resolution, input_size, *args, **kwargs)
        self._block_num = block_num
        
    @property
    def block_num(self):
        return self._block_num
    
    @abstractmethod
    def _forward_impl(self, *inputs, labels: Optional[torch.LongTensor]=None, start_block: int=None, end_block: int=None, **kwargs):
        pass
    
    def forward(self, *inputs, labels: Optional[torch.LongTensor]=None, start_block: Optional[int]=None, end_block: Optional[int]=None, **kwargs):
        
        if start_block is None:
            start_block = 0
            
        if end_block is None:
            end_block = self.block_num
            
        if start_block >= end_block:
            raise RuntimeError(f'expect `start_block` < `end_block`, but find `start_block`={start_block} and `end_block`={end_block}')
        
        return self._forward_impl(*inputs, labels, start_block, end_block, **kwargs)
