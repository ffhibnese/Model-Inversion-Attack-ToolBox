from copy import deepcopy

from torch import Tensor
from torch.nn import MaxPool2d
from ...utils import (
    BaseHook,
    FirstInputHook,
    OutputHook,
    DeepInversionBNFeatureHook,
    traverse_module,
)
from .base import *


class BaseClassifierWrapper(BaseImageClassifier):

    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
            *args,
            **kwargs,
        )

        self.module = module

    def preprocess_config_before_save(self, config):
        # return config
        process_config = {}
        for k, v in config:
            if k != 'module':
                process_config[k] = v

        config['module'] = {
            'model_name': CLASSNAME_TO_NAME_MAPPING[self.module.__class__.__name__],
            'config': self.module.preprocess_config_before_save(
                self.module._config_mixin_dict
            ),
        }

    @staticmethod
    def postprocess_config_after_load(config):
        config['module'] = auto_classifier_from_pretrained(config['module'])
        return config


@register_model('vib')
class VibWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
            *args,
            **kwargs,
        )

        # self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.st_layer = nn.Linear(self.hidden_dim, self.k * 2)
        # operate_fc(self.module, self.k * 2, None)
        self.fc_layer = nn.Linear(self.k, module.num_classes)

        self.feature_hook = FirstInputHook(self.fc_layer)

    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook

    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):

        # self._inner_hook.clear_feature()
        _, hook_res = self.module(image, *args, **kwargs)

        # # self._check_hook(HOOK_NAME_FEATURE)

        feature = hook_res[HOOK_NAME_FEATURE]

        statics = self.st_layer(feature)

        # statics, _ = self.module(image, *args, **kwargs)

        mu, std = statics[:, : self.k], statics[:, self.k : self.k * 2]

        self._last_statics = mu, std

        std = F.softplus(std - 5, beta=1)

        # eps = torch.FloatTensor(std.size()).normal_().to(std)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)

        return out, {'mu': mu, 'std': std}


def get_default_create_hidden_hook_fn(num: int = 3):

    param_num = num

    def _fn(model: BaseImageClassifier):
        linear_modules = []

        def _visit_fn(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                linear_modules.append(module)

        traverse_module(model, _visit_fn)
        linear_modules = linear_modules[1:]

        num = min(param_num, len(linear_modules))
        splitnum = (len(linear_modules) + 1) // (num + 1)
        use_nums = [splitnum * (i + 1) - 1 for i in range(num)]
        use_linear_modules = [linear_modules[i] for i in use_nums]
        return [FirstInputHook(l) for l in use_linear_modules]

    return _fn


def origin_vgg16_64_hidden_hook_fn(module):
    hiddens_hooks = []

    def _add_hook_fn(module):
        if isinstance(module, MaxPool2d):
            hiddens_hooks.append(OutputHook(module))

    traverse_module(module, _add_hook_fn, call_middle=False)
    return hiddens_hooks


@register_model('bido')
class BiDOWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        create_hidden_hook_fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
            *args,
            **kwargs,
        )

        # self.module = module

        create_hidden_hook_fn = (
            create_hidden_hook_fn
            if create_hidden_hook_fn is not None
            else get_default_create_hidden_hook_fn()
        )

        self.hidden_hooks = create_hidden_hook_fn(self.module)
        print(f'hidden hook num: {len(self.hidden_hooks)}')
        # exit()

    def get_last_feature_hook(self) -> BaseHook:
        return self.module.get_last_feature_hook()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        return forward_res, addition_info


def get_default_deepinversion_bn_hook_fn(num: int = 3):

    def _fn(model: BaseImageClassifier):
        bn_modules = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules.append(m)
        return [DeepInversionBNFeatureHook(l) for l in bn_modules]

    return _fn


class DeepInversionWrapper(BaseImageClassifier):

    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        create_bn_hook_fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
            *args,
            **kwargs,
        )

        self.module = module

        create_bn_hook_fn = (
            create_bn_hook_fn
            if create_bn_hook_fn is not None
            else get_default_deepinversion_bn_hook_fn()
        )

        self.bn_hooks = create_bn_hook_fn(module)

    def get_last_feature_hook(self) -> BaseHook:
        return self.module.get_last_feature_hook()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_DEEPINVERSION_BN] = [
            h.get_feature() for h in self.bn_hooks
        ]
        return forward_res, addition_info
