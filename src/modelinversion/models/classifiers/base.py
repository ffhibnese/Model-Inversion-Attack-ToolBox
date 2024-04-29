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

from ..utils import ModelMixin
from ...utils import traverse_name_module, FirstInputHook, BaseHook

HOOK_NAME_FEATURE = 'feature'
HOOK_NAME_HIDDEN = 'hidden'
HOOK_NAME_DEEPINVERSION_BN = 'deepinversion_bn'

BUILDIN_CLASSIFIERS = {}
CLASSNAME_TO_NAME_MAPPING = {}
TORCHVISION_MODEL_NAMES = tvmodel.list_models()


def register_model(name: Optional[str] = None):
    """Register model for construct.

    Args:
        name (Optional[str], optional): The key of the model. Defaults to None.
    """

    def wrapper(c):
        key = name if name is not None else c.__name__
        CLASSNAME_TO_NAME_MAPPING[c.__name__] = key
        if key in BUILDIN_CLASSIFIERS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILDIN_CLASSIFIERS[key] = c
        return c

    return wrapper


class ModelConstructException(Exception):
    pass


def construct_classifiers_by_name(name: str, **kwargs):

    if name in BUILDIN_CLASSIFIERS:
        return BUILDIN_CLASSIFIERS[name](**kwargs)

    if name in TORCHVISION_MODEL_NAMES:
        return TorchvisionClassifierModel(name, **kwargs)

    raise ModelConstructException(f'Module name {name} not found.')


def list_classifiers():
    """List all valid module names"""
    return sorted(BUILDIN_CLASSIFIERS.keys()) + TORCHVISION_MODEL_NAMES


def auto_classifier_from_pretrained(data_or_path):

    if isinstance(data_or_path, str):
        data = torch.load(data_or_path, map_location='cpu')
    else:
        data = data_or_path
    if 'model_name' not in data:
        raise RuntimeError('model_name is not contained in the data')

    cls: BaseImageClassifier = BUILDIN_CLASSIFIERS[data['model_name']]
    return cls.from_pretrained(data_or_path)


class BaseImageModel(ModelMixin):

    def __init__(self, resolution: int, feature_dim: int, **kwargs) -> None:
        nn.Module.__init__(self, **kwargs)

        self._resolution = resolution
        self._feature_dim = feature_dim
        self._inner_hooks = {}

    @property
    def resolution(self):
        return self._resolution

    @property
    def feature_dim(self):
        return self._feature_dim

    def _check_hook(self, name: str):
        if name not in self._inner_hooks:
            raise RuntimeError(f'The model do not have feature for `{name}`')

    def register_hook_for_forward(self, name: str, hook: BaseHook):
        self._inner_hooks[name] = hook

    @abstractmethod
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()

    def save_pretrained(self, path, **add_infos):
        return super().save_pretrained(
            path,
            model_name=CLASSNAME_TO_NAME_MAPPING[self.__class__.__name__],
            **add_infos,
        )

    def forward(self, image: torch.Tensor, *args, **kwargs):

        if image.shape[-1] != self.resolution or image.shape[-2] != self.resolution:
            image = TF.resize(image, (self.resolution, self.resolution), antialias=True)

        forward_res = self._forward_impl(image, *args, **kwargs)
        hook_res = {k: v.get_feature() for k, v in self._inner_hooks.items()}
        if isinstance(forward_res, tuple) and not isinstance(
            forward_res, InceptionOutputs
        ):
            if len(forward_res) != 2:
                raise RuntimeError(
                    f'The number of model output must be 1 or 2, but found {len(forward_res)}'
                )
            forward_res, forward_addition = forward_res
            if forward_addition is not None:
                for k, v in forward_addition.items():
                    if k in hook_res:
                        raise RuntimeError('hook result key conflict')
                    hook_res[k] = v
        return forward_res, hook_res


class BaseImageEncoder(BaseImageModel):

    def __init__(self, resolution: int, feature_dim: int, **kwargs) -> None:
        super().__init__(resolution, feature_dim, **kwargs)


class BaseImageClassifier(BaseImageModel):

    def __init__(
        self,
        resolution,
        feature_dim,
        num_classes,
        register_last_feature_hook=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(resolution, feature_dim, *args, **kwargs)
        self._num_classes = num_classes

        self._feature_flag = False

        self.register_last_feature_hook = register_last_feature_hook

    @property
    def num_classes(self):
        return self._num_classes

    def get_last_feature_hook(self) -> BaseHook:
        return None

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['register_last_feature_hook']
        return super().preprocess_config_before_save(config)

    def forward(self, image: torch.Tensor, *args, **kwargs):
        if not self._feature_flag and self.register_last_feature_hook:
            self._feature_flag = True
            hook = self.get_last_feature_hook()
            if hook is None:
                raise RuntimeError('The last feature hook is not set.')
            self.register_hook_for_forward(HOOK_NAME_FEATURE, hook=hook)
        return super().forward(image, *args, **kwargs)


def _operate_fc_impl(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
):
    """Reset the output class num of nn.Linear and return the input feature_dim of nn.Linear.

    Args:
        module (nn.Module): The specific model structure.
        reset_num_classes (int, optional): The new output class num. Defaults to None.
        visit_fc_fn (Callable, optional): Other operations to the nn.Linear of the input module. Defaults to None.

    Returns:
        feature_dim (int): The input feature_dim of nn.Linear.
    """

    if isinstance(module, nn.Sequential):

        if len(module) == 0:
            raise ModelConstructException('fail to implement')

        if isinstance(module[-1], nn.Linear):
            feature_dim = module[-1].weight.shape[-1]

            if (
                reset_num_classes is not None
                and reset_num_classes != module[-1].weight.shape[0]
            ):
                module[-1] = nn.Linear(feature_dim, reset_num_classes)

            if visit_fc_fn is not None:
                visit_fc_fn(module[-1])

            return feature_dim
        else:
            return _operate_fc_impl(module[-1], reset_num_classes)

    children = list(module.named_children())
    if len(children) == 0:
        raise ModelConstructException('fail to implement')
    attr_name, child_module = children[-1]
    if isinstance(child_module, nn.Linear):
        feature_dim = child_module.weight.shape[-1]

        if (
            reset_num_classes is not None
            and reset_num_classes != child_module.weight.shape[0]
        ):
            setattr(module, attr_name, nn.Linear(feature_dim, reset_num_classes))

        if visit_fc_fn is not None:
            visit_fc_fn(getattr(module, attr_name))

        return feature_dim
    else:
        return _operate_fc_impl(child_module, reset_num_classes)


def operate_fc(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
) -> int:
    return _operate_fc_impl(module, reset_num_classes, visit_fc_fn)


@register_model('torchvision')
class TorchvisionClassifierModel(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        resolution=224,
        weights=None,
        arch_kwargs={},
        register_last_feature_hook=False,
    ) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'

        self._feature_hook = None

        def _add_hook_fn(m):
            self._feature_hook = FirstInputHook(m)

        tv_module = importlib.import_module('torchvision.models')
        factory = getattr(tv_module, arch_name, None)
        if factory is None:
            raise RuntimeError(f'torchvision do not support model {arch_name}')
        model = factory(weights=weights, **arch_kwargs)

        feature_dim = operate_fc(model, num_classes, _add_hook_fn)

        super().__init__(
            resolution, feature_dim, num_classes, register_last_feature_hook
        )

        self.model = model

    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)

    def get_last_feature_hook(self) -> BaseHook:
        return self._feature_hook
