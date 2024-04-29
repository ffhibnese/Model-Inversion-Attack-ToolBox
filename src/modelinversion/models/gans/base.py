from abc import abstractmethod
from copy import deepcopy
from typing import Union, Optional, Sequence, Callable

import torch
import torch.nn as nn

from ..utils import ModelMixin


class _BUILDIN_INFO:
    """Info Class to management models"""

    def __init__(self) -> None:
        self._buildin_dict = {}
        self._alias2name = {}
        self._name2alias = {}

    def register(self, fn: Callable, name: str, alias: list[str] = None):

        use_names = [name]
        if alias is not None:
            use_names += alias

        for use_name in use_names:
            if use_name in self._alias2name or use_name in self._buildin_dict:
                raise ValueError(
                    f"An entry is already registered under the name '{use_name}'."
                )

        self._buildin_dict[name] = fn
        if alias is not None:
            for alia in alias:
                self._alias2name[alia] = name
            self._name2alias[name] = alias

    def get_builder(self, name: str):

        if name in self._alias2name:
            key = self._alias2name[name]
        else:
            key = name
        if key not in self._buildin_dict:
            raise GanConstructException(f'GAN name {name} not found.')

        return self._buildin_dict[key]

    def list_items(self, alias: bool = False):

        ret = sorted(self._buildin_dict.keys())
        if alias:
            ret += sorted(self._alias2name.keys())

        return ret

    def show_items(self):

        names = sorted(self._buildin_dict.keys())

        for name in names:
            print(names, end=' ')
            if name in self._name2alias:
                join_res = ', '.join(self._name2alias(name))
                print(f'({join_res})', end='')
            print()


BUILTIN_GENERATORS = _BUILDIN_INFO()
BUILTIN_DISCRIMINATORS = _BUILDIN_INFO()

GENERATOR_CLASSNAME_TO_NAME_MAPPTING = {}
DISCRIMINATOR_CLASSNAME_TO_NAME_MAPPTING = {}


def _register_model(
    buildin_info: _BUILDIN_INFO,
    mapping,
    name: Optional[str] = None,
    alias: list[str] = None,
):
    """Register model.

    Args:
        name (Optional[str], optional): The key of the model. Defaults to None.
    """

    def wrapper(fn):
        key = name if name is not None else fn.__name__
        mapping[fn.__name__] = name
        buildin_info.register(fn, name=key, alias=alias)
        return fn

    return wrapper


def register_generator(name: Optional[str] = None, alias: Optional[list[str]] = None):
    """Register generator.

    Args:
        name (Optional[str], optional): The key of the generator. Defaults to None.
        alias (Optional[list[str]], optional): The alias of the generator.
    """
    return _register_model(
        BUILTIN_GENERATORS, GENERATOR_CLASSNAME_TO_NAME_MAPPTING, name=name, alias=alias
    )


def register_discriminator(
    name: Optional[str] = None, alias: Optional[list[str]] = None
):
    """Register discriminator.

    Args:
        name (Optional[str], optional): The key of the discriminator. Defaults to None.
        alias (Optional[list[str]], optional): The alias of the discriminator.
    """
    return _register_model(
        BUILTIN_DISCRIMINATORS,
        DISCRIMINATOR_CLASSNAME_TO_NAME_MAPPTING,
        name=name,
        alias=alias,
    )


def _construct_model_by_name(buildin_info: _BUILDIN_INFO, name: str, **kwargs):

    builder = buildin_info.get_builder(name)
    return builder(**kwargs)


def construct_generator_by_name(name: str, **kwargs):
    return _construct_model_by_name(BUILTIN_GENERATORS, name=name, **kwargs)


def construct_discriminator_by_name(name: str, **kwargs):
    return _construct_model_by_name(BUILTIN_DISCRIMINATORS, name=name, **kwargs)


def _list_models(buildin_info: _BUILDIN_INFO, alias=False):
    return buildin_info.list_items(alias=alias)


def list_generators(alias=False):
    return _list_models(BUILTIN_GENERATORS, alias=alias)


def list_discriminators(alias=False):
    return _list_models(BUILTIN_DISCRIMINATORS, alias=alias)


def _show_models(buildin_info: _BUILDIN_INFO):
    return buildin_info.show_items()


def show_generators():
    return _show_models(BUILTIN_GENERATORS)


def show_discriminators():
    return _show_models(BUILTIN_DISCRIMINATORS)


class GanConstructException(Exception):
    pass


class LambdaModule(nn.Module):

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BaseImageGenerator(ModelMixin):

    def __init__(self, resolution, input_size: Union[int, Sequence[int]]) -> None:
        super().__init__()

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

    def save_pretrained(self, path, **add_infos):
        return super().save_pretrained(
            path,
            model_name=GENERATOR_CLASSNAME_TO_NAME_MAPPTING[self.__class__.__name__],
            **add_infos,
        )

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


class BaseDiscriminator(ModelMixin):

    def __init__(self) -> None:
        super().__init__()

    def save_pretrained(self, path, **add_infos):
        return super().save_pretrained(
            path,
            model_name=DISCRIMINATOR_CLASSNAME_TO_NAME_MAPPTING[
                self.__class__.__name__
            ],
            **add_infos,
        )


def _auto_model_from_pretrained(buildin_info: _BUILDIN_INFO, data_or_path: dict | str):

    if isinstance(data_or_path, str):
        data = torch.load(data_or_path, map_location='cpu')
    else:
        data = data_or_path
    if 'model_name' not in data:
        raise RuntimeError('model_name is not contained in the data')
    cls: ModelMixin = buildin_info.get_builder(data['model_name'])
    return cls.from_pretrained(data_or_path)


def auto_generator_from_pretrained(data_or_path: dict | str):
    return _auto_model_from_pretrained(BUILTIN_GENERATORS, data_or_path)


def auto_discriminator_from_pretrained(data_or_path: dict | str):
    return _auto_model_from_pretrained(BUILTIN_DISCRIMINATORS, data_or_path)
