from abc import abstractmethod
from typing import Union, Optional, Sequence

import torch
import torch.nn as nn

from ...utils import unwrapped_parallel_module


class LambdaModule(nn.Module):

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BaseImageGenerator(nn.Module):

    def __init__(
        self, resolution, input_size: Union[int, Sequence[int]], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._resolution = resolution
        if isinstance(input_size, int):
            input_size = (input_size,)
        if not isinstance(input_size, tuple):
            input_size = tuple(input_size)
        self._input_size = input_size

    @property
    def resolution(self):
        return self._resolution

    @property
    def input_size(self):
        return self._input_size

    # @staticmethod
    # def sample_latents(self, sample_num: int, batch_size: int, **kwargs):
    #     size = (sample_num, ) + unwrapped_parallel_module(self).input_size
    #     return torch.randn(size)

    @abstractmethod
    def _forward_impl(self, *inputs, labels=None, **kwargs):
        raise NotImplementedError()

    def forward(self, *inputs, labels=None, **kwargs):
        return self._forward_impl(*inputs, labels, **kwargs)


class BaseIntermediateImageGenerator(BaseImageGenerator):

    def __init__(
        self, resolution, input_size: int | tuple[int], block_num, *args, **kwargs
    ) -> None:
        super().__init__(resolution, input_size, *args, **kwargs)
        self._block_num = block_num

    @property
    def block_num(self):
        return self._block_num

    @abstractmethod
    def _forward_impl(
        self,
        *inputs,
        labels: Optional[torch.LongTensor],
        start_block: int,
        end_block: int,
        **kwargs,
    ):
        raise NotImplementedError()

    def forward(
        self,
        *inputs,
        labels: Optional[torch.LongTensor] = None,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        **kwargs,
    ):

        if start_block is None:
            start_block = 0

        if end_block is None:
            end_block = self.block_num

        if start_block >= end_block:
            raise RuntimeError(
                f'expect `start_block` < `end_block`, but find `start_block`={start_block} and `end_block`={end_block}'
            )

        return self._forward_impl(
            *inputs,
            labels=labels,
            start_block=start_block,
            end_block=end_block,
            **kwargs,
        )
