from copy import deepcopy
from collections import OrderedDict

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
from ..layers import DynamicConv2D


class BaseClassifierWrapper(BaseImageClassifier):

    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
        )

        self.module = module

    def preprocess_config_before_save(self, config):
        # return config
        process_config = {}
        for k, v in config.items():
            if k != 'module':
                process_config[k] = v

        config['module'] = {
            'model_name': CLASSNAME_TO_NAME_MAPPING[self.module.__class__.__name__],
            'config': self.module.preprocess_config_before_save(
                self.module._config_mixin_dict
            ),
        }

        return super().preprocess_config_before_save(config)

    @staticmethod
    def postprocess_config_after_load(config):
        config['module'] = auto_classifier_from_pretrained(config['module'])
        return config


@register_model('ztq')
class ZtqDefenseWrapper(BaseImageClassifier):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        binary_checker: BaseImageClassifier,
        register_last_feature_hook=False,
    ):
        super().__init__(
            classifier.resolution,
            classifier.feature_dim,
            classifier.num_classes,
            register_last_feature_hook,
        )
        self.classifier = classifier
        self.binary_checker = binary_checker

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        out, info = self.classifier(image, *args, **kwargs)

        # 这里修改 out

        return out, info

    def preprocess_config_before_save(self, config):
        # return config
        process_config = {}
        for k, v in config.items():
            if k != '_classifier' and k != '_binary_checker':
                process_config[k] = v

        config['_classifier'] = {
            'model_name': CLASSNAME_TO_NAME_MAPPING[self.classifier.__class__.__name__],
            'config': self.classifier.preprocess_config_before_save(
                self.classifier._config_mixin_dict
            ),
        }

        config['_binary_checker'] = {
            'model_name': CLASSNAME_TO_NAME_MAPPING[
                self.binary_checker.__class__.__name__
            ],
            'config': self.binary_checker.preprocess_config_before_save(
                self.binary_checker._config_mixin_dict
            ),
        }

        return super().preprocess_config_before_save(config)

    @staticmethod
    def postprocess_config_after_load(config):
        config['_classifier'] = auto_classifier_from_pretrained(config['_classifier'])
        config['_binary_checker'] = auto_classifier_from_pretrained(
            config['_binary_checker']
        )
        return config


@register_model('vib')
class VibWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
        )

        # self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.st_layer = nn.Linear(self.hidden_dim, self.k * 2)
        # operate_fc(self.module, self.k * 2, None)
        self.fc_layer = nn.Linear(self.k, module.num_classes)

        # self.feature_hook = FirstInputHook(self.fc_layer)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.feature_hook

    @staticmethod
    def postprocess_config_after_load(config):
        config['module'] = auto_classifier_from_pretrained(
            config['module'], register_last_feature_hook=True
        )
        return config

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

        return out, {'mu': mu, 'std': std, HOOK_NAME_FEATURE: feat}


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
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
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

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.module.get_last_feature_hook()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        return forward_res, addition_info


@register_model('lora')
class LoraWrapper(BaseClassifierWrapper):

    def _get_split_idx(self, length, ratio):
        if ratio == 0:
            return 0
        if isinstance(ratio, int):
            return length // ratio
        if 0 < ratio < 1:
            return int(length * ratio)
        raise RuntimeError(f'ratio {ratio} is invalid.')

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        # create_hidden_hook_fn: Optional[Callable] = None,
        lora_dim=5,
        start_ratio=3,
        end_ratio=1,
        lora_step=1,
        a_k=0,
        b_k=0,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
        )

        optim_nodes = nn.ModuleList()

        lins: list[nn.Linear] = []
        convs: list[nn.Conv2d] = []

        def _visit_linear(module):
            if isinstance(module, nn.Linear):
                lins.append(module)
            elif isinstance(module, nn.Conv2d):
                convs.append(module)

        traverse_module(module, _visit_linear, call_middle=False)

        start_idx = self._get_split_idx(len(convs), start_ratio)
        end_idx = self._get_split_idx(len(convs), end_ratio)

        # convs = convs[len(convs) // 3 :]
        node_a_cls = nn.Conv2d if a_k == 0 else DynamicConv2D
        node_b_cls = nn.Conv2d if b_k == 0 else DynamicConv2D
        lora_idx = 0
        for i, conv in enumerate(convs[start_idx:end_idx]):
            if i % lora_step != 0:
                continue
            if a_k == 0:
                node_a = nn.Conv2d(
                    conv.in_channels,
                    lora_dim,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=False,
                    padding_mode=conv.padding_mode,
                )
            else:
                # print(conv.groups)
                # exit()
                node_a = DynamicConv2D(
                    conv.in_channels,
                    lora_dim,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=False,
                    # padding_mode=conv.padding_mode,
                    K=a_k,
                )
            if b_k == 0:
                node_b = nn.Conv2d(
                    lora_dim, conv.out_channels, kernel_size=1, bias=conv.bias
                )
            else:
                node_b = DynamicConv2D(
                    lora_dim, conv.out_channels, kernel_size=1, bias=conv.bias, K=b_k
                )
            nn.init.zeros_(node_b.weight)

            if node_b.bias is not None:
                nn.init.zeros_(node_b.bias)

            optim_nodes.append(node_a)
            optim_nodes.append(node_b)
            conv._lora_idx = lora_idx

            def hook_fn(module, inp, oup):
                lora_idx = module._lora_idx
                node_a = optim_nodes[2 * lora_idx]
                node_b = optim_nodes[2 * lora_idx + 1]
                a_out = node_a(inp[0])
                b_out = node_b(a_out)

                return b_out + oup

            conv.register_forward_hook(hook_fn)

            lora_idx += 1

        # lins = lins[:-1]
        print('add lora num: ', len(optim_nodes))

        for i, conv in enumerate(convs[end_idx:]):
            optim_nodes.append(conv)

        print(f'full tune num: ', len(convs) - end_idx)

        optim_nodes.append(lins[-1])

        self.optim_nodes = optim_nodes

    def freeze_to_train(self):

        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.optim_nodes.parameters():
            p.requires_grad_(True)

    def unwrap(self) -> BaseImageClassifier:
        model = deepcopy(self.module)

        def _visit(module):
            if isinstance(module, nn.Conv2d) and hasattr(module, '_lora_idx'):
                idx = module._lora_idx
                del module._lora_idx
                conv1 = self.optim_nodes[2 * idx]
                conv2 = self.optim_nodes[2 * idx + 1]

                combined_weight = torch.matmul(
                    conv2.weight.view(conv2.out_channels, -1),
                    conv1.weight.view(conv1.out_channels, -1),
                ).view(conv2.out_channels, conv1.in_channels, *conv1.kernel_size)

                module.weight.data.add_(combined_weight.data)
                if conv2.bias is not None:
                    module.bias.data.add_(conv2.bias.data)
                module._forward_hooks = OrderedDict()

        traverse_module(model, _visit, call_middle=False)

        return model

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        return forward_res, addition_info


@register_model('growlora')
class GrowLoraWrapper(BaseClassifierWrapper):

    def _get_split_idx(self, length, ratio):
        if ratio == 0:
            return 0
        if isinstance(ratio, int):
            return length // ratio
        if 0 < ratio < 1:
            return int(length * ratio)
        raise RuntimeError(f'ratio {ratio} is invalid.')

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        # create_hidden_hook_fn: Optional[Callable] = None,
        start_lora_dim=3,
        end_lora_dim=8,
        start_ratio=3,
        end_ratio=1,
        lora_step=1,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
        )

        optim_nodes = nn.ModuleList()

        lins: list[nn.Linear] = []
        convs: list[nn.Conv2d] = []

        def _visit_linear(module):
            if isinstance(module, nn.Linear):
                lins.append(module)
            elif isinstance(module, nn.Conv2d):
                convs.append(module)

        traverse_module(module, _visit_linear, call_middle=False)

        start_idx = self._get_split_idx(len(convs), start_ratio)
        end_idx = self._get_split_idx(len(convs), end_ratio)
        end_lora_dim += 1

        # convs = convs[len(convs) // 3 :]
        lora_idx = 0
        for i, conv in enumerate(convs[start_idx:end_idx]):
            lora_dim = start_lora_dim + int(
                (i) * (end_lora_dim - start_lora_dim) // (end_idx - start_idx)
            )
            if i % lora_step != 0:
                continue
            node_a = nn.Conv2d(
                conv.in_channels,
                lora_dim,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=False,
                padding_mode=conv.padding_mode,
            )
            node_b = nn.Conv2d(
                lora_dim, conv.out_channels, kernel_size=1, bias=conv.bias
            )
            nn.init.zeros_(node_b.weight)
            # print(conv.in_channels, lora_dim, conv.out_channels, end=' | ')

            if node_b.bias is not None:
                nn.init.zeros_(node_b.bias)

            optim_nodes.append(node_a)
            optim_nodes.append(node_b)
            conv._lora_idx = lora_idx

            def hook_fn(module, inp, oup):
                lora_idx = module._lora_idx
                node_a = optim_nodes[2 * lora_idx]
                node_b = optim_nodes[2 * lora_idx + 1]
                a_out = node_a(inp[0])
                b_out = node_b(a_out)

                return b_out + oup

            conv.register_forward_hook(hook_fn)

            lora_idx += 1

        # lins = lins[:-1]
        print('add lora num: ', len(optim_nodes))
        # exit()

        for i, conv in enumerate(convs[end_idx:]):
            optim_nodes.append(conv)

        print(f'full tune num: ', len(convs) - end_idx)

        optim_nodes.append(lins[-1])

        self.optim_nodes = optim_nodes

    def freeze_to_train(self):

        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.optim_nodes.parameters():
            p.requires_grad_(True)

    def unwrap(self) -> BaseImageClassifier:
        model = deepcopy(self.module)

        def _visit(module):
            if isinstance(module, nn.Conv2d) and hasattr(module, '_lora_idx'):
                idx = module._lora_idx
                del module._lora_idx
                conv1 = self.optim_nodes[2 * idx]
                conv2 = self.optim_nodes[2 * idx + 1]

                combined_weight = torch.matmul(
                    conv2.weight.view(conv2.out_channels, -1),
                    conv1.weight.view(conv1.out_channels, -1),
                ).view(conv2.out_channels, conv1.in_channels, *conv1.kernel_size)

                module.weight.data.add_(combined_weight.data)
                if conv2.bias is not None:
                    module.bias.data.add_(conv2.bias.data)
                module._forward_hooks = OrderedDict()

        traverse_module(model, _visit, call_middle=False)

        return model

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
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
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
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
