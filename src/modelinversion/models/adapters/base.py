import importlib
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Optional, Any
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodel
import torchvision.transforms.functional as TF
from torchvision.models.inception import InceptionOutputs

from ..base import ModelMixin
from ...utils import traverse_name_module, FirstInputHook, BaseHook

BUILDIN_ADAPTERS = {}
CLASSNAME_TO_NAME_MAPPING = {}


def register_adapter(name: Optional[str] = None):
    """Register model for construct.

    Args:
        name (Optional[str], optional): The key of the model. Defaults to None.
    """

    def wrapper(c):
        key = name if name is not None else c.__name__
        CLASSNAME_TO_NAME_MAPPING[c.__name__] = key
        if key in BUILDIN_ADAPTERS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILDIN_ADAPTERS[key] = c
        return c

    return wrapper


class BaseAdapter(ModelMixin):

    def save_pretrained(self, path, **add_infos):
        return super().save_pretrained(
            path,
            model_name=CLASSNAME_TO_NAME_MAPPING[self.__class__.__name__],
            **add_infos,
        )


class ModelConstructException(Exception):
    pass


def construct_adapters_by_name(name: str, **kwargs):

    if name in BUILDIN_ADAPTERS:
        return BUILDIN_ADAPTERS[name](**kwargs)

    raise ModelConstructException(f'Module name {name} not found.')


def list_adapters():
    """List all valid module names"""
    return sorted(BUILDIN_ADAPTERS.keys())


def auto_adapter_from_pretrained(data_or_path, **kwargs):

    if isinstance(data_or_path, str):
        data = torch.load(data_or_path, map_location='cpu')
    else:
        data = data_or_path
    if 'model_name' not in data:
        raise RuntimeError('model_name is not contained in the data')

    cls: ModelMixin = BUILDIN_ADAPTERS[data['model_name']]
    return cls.from_pretrained(data_or_path, **kwargs)
