from copy import deepcopy
from collections import OrderedDict
from typing import Iterator

from torch import Tensor
from torch.nn import MaxPool2d
from torch.nn.parameter import Parameter
from torchvision.models.swin_transformer import SwinTransformerBlock
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


_activation = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'none': nn.Identity,
}


def _neck_builder(neck_dim, activation='tanh'):

    activation_builder = _activation[activation]

    def _builder(input_dim, output_dim):
        return nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, neck_dim),
            activation_builder(),
            nn.Linear(neck_dim, output_dim),
        )

    return _builder


@register_model('neck')
class NeckWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=True,
        neck_dim=10,
        neck_activation='tanh',
        feature_compressed=False,
    ) -> None:

        def _output_transform(m: nn.Sequential):
            # self._feature_hook = FirstInputHook(m)
            def hook_fn(module, input, output):
                # print(type(input))
                # print(type(input[0]))
                # print(type(output))
                # print(type(output[0]))
                # exit()
                return output, {HOOK_NAME_FEATURE: input[0]}

            # print('hook register')
            if feature_compressed:
                m[-1].register_forward_hook(hook_fn)
            else:
                m.register_forward_hook(hook_fn)

        operate_fc(
            module,
            module.num_classes,
            _output_transform,
            _neck_builder(neck_dim=neck_dim, activation=neck_activation),
        )

        # self.module = module

        super().__init__(
            module,
            register_last_feature_hook,
        )

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        result = self.module(image, *args, **kwargs)
        if isinstance(result, tuple):
            result, addition_info = result
            addition_info: dict
            if isinstance(result, tuple):
                result, new_addition_info = result
                addition_info.update(new_addition_info)
        else:
            addition_info = {}
        # print(type(result))
        # print(type(result[0]))
        # print(type(result[0][0]))
        # exit()
        return result, addition_info


# nn.modules.activation.__all__


def recurrent_replace_activation(module, activation='tanh'):

    replace_num = 0
    if isinstance(module, nn.Sequential):
        for i, m in enumerate(module):
            if m.__class__.__name__ in nn.modules.activation.__all__:
                module[i] = _activation[activation]()
                replace_num += 1
            else:
                replace_num += recurrent_replace_activation(m, activation)[1]
        return module, replace_num

    for name, m in module.named_children():
        if m.__class__.__name__ in nn.modules.activation.__all__:
            setattr(module, name, _activation[activation]())
            replace_num += 1
        else:
            replace_num += recurrent_replace_activation(m, activation)[1]
    return module, replace_num


@register_model('activation_replacer')
class ActivationReplacerWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=True,
        activation='relu',
    ) -> None:

        # replace every activation function in module with the input activation
        module, replace_num = recurrent_replace_activation(module, activation)

        print(replace_num)

        super().__init__(
            module,
            register_last_feature_hook,
        )

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        return self.module(image, *args, **kwargs)


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
            if isinstance(module, (nn.Conv2d, SwinTransformerBlock)):
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

    def unwrap(self):
        return self.module

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

        lora_idx = 0

        lora_step_cnt = lora_step - 1
        for i, conv in enumerate(convs[start_idx:end_idx]):
            # if i % lora_step != 0:
            #     continue
            lora_step_cnt += 1
            if lora_step_cnt >= lora_step:
                lora_step_cnt -= lora_step
            else:
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

        self.freeze_to_train()

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     print('get optim nodes parameters')
    #     return self.optim_nodes.parameters(recurse)

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

        self.freeze_to_train()

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     print('get optim nodes parameters')
    #     return self.optim_nodes.parameters(recurse)

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


@register_model('lrc')
class LRCWrapper(BaseClassifierWrapper):

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
        keep_rank_or_ratio: int | float | list[int] = 1.0,
        start_ratio=0,
        end_ratio=1,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.keep_rank_or_ratio = keep_rank_or_ratio
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        self.compression()

        # self.lins = lins[-1]
        # self.convs = convs[-1]

    def compression(self):

        lins: list[nn.Linear] = []
        convs: list[nn.Conv2d] = []

        def _visit_compression(module):
            if isinstance(module, nn.Linear):
                lins.append(module)
            elif isinstance(module, nn.Conv2d):
                convs.append(module)

        traverse_module(self.module, _visit_compression, call_middle=False)

        lins = lins[:-1]

        conv_start_idx = self._get_split_idx(len(convs), self.start_ratio)
        conv_end_idx = self._get_split_idx(len(convs), self.end_ratio)

        lin_start_idx = self._get_split_idx(len(lins), self.start_ratio)
        lin_end_idx = self._get_split_idx(len(lins), self.end_ratio)

        for lin in lins[lin_start_idx:lin_end_idx]:
            self._split_linear(lin, self.keep_rank_or_ratio)

        for conv in convs[conv_start_idx:conv_end_idx]:
            self._split_conv(conv, self.keep_rank_or_ratio)

    @torch.no_grad()
    def _split_conv(self, conv: nn.Conv2d, keep_rank_or_ratio):

        # (O, I, K, K)
        original_weight = conv.weight.data
        # (O, I * K * K)
        reshaped_weight = original_weight.reshape(original_weight.size(0), -1)

        # hw = original_weight.shape[-1] * original_weight.shape[-2]

        full_rank = min(reshaped_weight.shape[1], reshaped_weight.shape[0])

        # (O, full_rank), (full_rank), (I * K * K, full_rank)
        U, S, V = torch.svd(reshaped_weight)
        if isinstance(keep_rank_or_ratio, float):
            if keep_rank_or_ratio == 1.0:
                # print("AA")
                k = full_rank
            elif keep_rank_or_ratio == 0.0:
                k = 0
            else:
                # print("BB")
                # cumulative_energy = torch.cumsum(S, dim=0) / torch.sum(S)
                # k = (
                #     torch.searchsorted(
                #         cumulative_energy, torch.tensor(keep_rank_or_ratio)
                #     ).item()
                #     + 1
                # )
                k = int(full_rank * keep_rank_or_ratio)
                # k = (k - 1) // hw + 1
                k = min(k, full_rank)
        else:
            # print("CC")
            k = keep_rank_or_ratio

        # select_dim = k * hw

        # (O, k), (k), (I * K * K, k)
        U_k = U[:, :k]
        S_k = torch.diag(S[:k])
        V_k = V[:, :k]

        print(f"origin dim: {full_rank} new dim: {k}")

        node_a = nn.Conv2d(
            conv.in_channels,
            k,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False,
            padding_mode=conv.padding_mode,
        )
        node_b = nn.Conv2d(
            k, conv.out_channels, kernel_size=1, bias=conv.bias is not None
        )

        # print(U.shape, S.shape, V.shape)
        # print(U_k.shape, S_k.shape, V_k.shape)
        # print(conv.weight.shape, node_a.weight.shape, node_b.weight.shape)
        # # exit()

        # print(
        #     f"V_K.T shape: {V_k.T.shape}",
        #     " ",
        #     conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1],
        #     "node_a shape: ",
        #     node_a.weight.shape,
        #     # V_k.T.view(k, conv.in_channels, *conv.kernel_size).shape,
        # )

        node_a.weight.data = V_k.T.reshape(
            k, conv.in_channels, conv.kernel_size[0], conv.kernel_size[1]
        )
        node_b.weight.data = (U_k @ S_k).reshape(conv.out_channels, k, 1, 1)
        if conv.bias is not None:
            node_b.bias.data = conv.bias

        # return nn.Sequential(node_a, node_b)
        conv._lrc_node_a = node_a
        conv._lrc_node_b = node_b
        conv._lrc_save_require_grad = conv.weight.requires_grad
        conv.requires_grad_(False)

        def _new_forward(self, x):
            a_out = self._lrc_node_a(x)
            b_out = self._lrc_node_b(a_out)
            return b_out

        conv._lrc_save_forward = conv.forward

        conv.forward = _new_forward.__get__(conv, conv.__class__)

    # @torch.no_grad()
    # def _split_conv(self, conv: nn.Conv2d, keep_rank_or_ratio):

    #     # (O, I, K, K)
    #     original_weight = conv.weight.data
    #     # (O, I * K * K)
    #     reshaped_weight = original_weight.reshape(original_weight.size(0), -1)

    #     # (I*K*K, O)
    #     transpose_weight = reshaped_weight.transpose(0, 1)

    #     hw = original_weight.shape[-1] * original_weight.shape[-2]

    #     # (I*K*K, O), (O), (O, O)
    #     U, S, V = torch.svd(transpose_weight)
    #     if isinstance(keep_rank_or_ratio, float):
    #         if keep_rank_or_ratio == 1.0:
    #             # print("AA")
    #             k = reshaped_weight.shape[1]
    #         elif keep_rank_or_ratio == 0.0:
    #             k = 0
    #         else:
    #             # print("BB")
    #             cumulative_energy = torch.cumsum(S, dim=0) / torch.sum(S)
    #             k = (
    #                 torch.searchsorted(
    #                     cumulative_energy, torch.tensor(keep_rank_or_ratio)
    #                 ).item()
    #                 + 1
    #             )
    #             # k = (k - 1) // hw + 1
    #             k = min(k, reshaped_weight.shape[1])
    #     else:
    #         # print("CC")
    #         k = keep_rank_or_ratio * hw

    #     # select_dim = k * hw

    #     # (I*K*K, k), (k), (O, k)
    #     U_k = U[:, :k]
    #     S_k = torch.diag(S[:k])
    #     V_k = V[:, :k]

    #     print(f"origin dim: {original_weight.shape[1]} new dim: {k}")

    #     node_a = nn.Conv2d(
    #         conv.in_channels,
    #         k,
    #         kernel_size=conv.kernel_size,
    #         stride=conv.stride,
    #         padding=conv.padding,
    #         dilation=conv.dilation,
    #         groups=conv.groups,
    #         bias=False,
    #         padding_mode=conv.padding_mode,
    #     )
    #     node_b = nn.Conv2d(
    #         k, conv.out_channels, kernel_size=1, bias=conv.bias is not None
    #     )

    #     print(U.shape, S.shape, V.shape)
    #     print(U_k.shape, S_k.shape, V_k.shape)
    #     print(conv.weight.shape, node_a.weight.shape, node_b.weight.shape)
    #     # exit()

    #     print(
    #         f"V_K.T shape: {V_k.T.shape}",
    #         " ",
    #         conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1],
    #         "node_a shape: ",
    #         node_a.weight.shape,
    #         # V_k.T.view(k, conv.in_channels, *conv.kernel_size).shape,
    #     )

    #     node_a.weight.data = (U_k @ S_k).reshape(
    #         k, conv.in_channels, conv.kernel_size[0], conv.kernel_size[1]
    #     )
    #     node_b.weight.data = V_k.T.reshape(conv.out_channels, k, 1, 1)
    #     if conv.bias is not None:
    #         node_b.bias.data = conv.bias

    #     # return nn.Sequential(node_a, node_b)
    #     conv._lrc_node_a = node_a
    #     conv._lrc_node_b = node_b
    #     conv._lrc_save_require_grad = conv.weight.requires_grad
    #     conv.requires_grad_(False)

    #     def _new_forward(self, x):
    #         a_out = self._lrc_node_a(x)
    #         b_out = self._lrc_node_b(a_out)
    #         return b_out

    #     conv._lrc_save_forward = conv.forward

    #     conv.forward = _new_forward.__get__(conv, conv.__class__)

    def _split_linear(self, linear: nn.Linear, keep_rank_or_ratio):

        # (O. I)
        original_weight = linear.weight.data
        # (O, I), (I), (I, I)
        U, S, V = torch.svd(original_weight)
        full_rank = min(original_weight.shape[0], original_weight.shape[1])
        if isinstance(keep_rank_or_ratio, float):
            # cumulative_energy = torch.cumsum(S, dim=0) / torch.sum(S)
            # k = (
            #     torch.searchsorted(
            #         cumulative_energy, torch.tensor(keep_rank_or_ratio)
            #     ).item()
            #     + 1
            # )

            k = int(full_rank * keep_rank_or_ratio)
            k = min(k, len(S))
        else:
            k = keep_rank_or_ratio
        # (O, k), (k), (k, I)
        U_k = U[:, :k]
        S_k = torch.diag(S[:k])
        V_k = V[:, :k]
        node_a = nn.Linear(linear.in_features, k, bias=False)
        node_b = nn.Linear(k, linear.out_features, bias=linear.bias is not None)

        node_a.weight.data = V_k.T
        node_b.weight.data = U_k @ S_k
        if linear.bias is not None:
            node_b.bias.data = linear.bias

        linear._lrc_node_a = node_a
        linear._lrc_node_b = node_b
        linear._lrc_save_require_grad = linear.weight.requires_grad
        linear.requires_grad_(False)

        def _new_forward(self, x):
            a_out = self._lrc_node_a(x)
            b_out = self._lrc_node_b(a_out)
            return b_out

        linear._lrc_save_forward = linear.forward

        linear.forward = _new_forward.__get__(linear, linear.__class__)

        return nn.Sequential(node_a, node_b)

    def unwrap(self) -> BaseImageClassifier:
        model = deepcopy(self.module)

        def _visit(module):
            if isinstance(module, nn.Conv2d) and hasattr(module, '_lrc_node_a'):
                conv1 = module._lrc_node_a
                conv2 = module._lrc_node_b

                combined_weight = torch.matmul(
                    conv2.weight.view(conv2.out_channels, -1),
                    conv1.weight.view(conv1.out_channels, -1),
                ).view(conv2.out_channels, conv1.in_channels, *conv1.kernel_size)

                module.weight.data = combined_weight.data
                if conv2.bias is not None:
                    module.bias.data = conv2.bias.data

                del module._lrc_node_a, module._lrc_node_b

                module.forward = module._lrc_save_forward
                module.requires_grad_(module._lrc_save_require_grad)

            if isinstance(module, nn.Linear) and hasattr(module, '_lrc_node_a'):
                linear1 = module._lrc_node_a
                linear2 = module._lrc_node_b

                combined_weight = torch.matmul(linear2.weight, linear1.weight)

                module.weight.data = combined_weight.data
                if linear2.bias is not None:
                    module.bias.data = linear2.bias.data
                del module._lrc_node_a, module._lrc_node_b
                module.forward = module._lrc_save_forward
                module.requires_grad_(module._lrc_save_require_grad)

        traverse_module(model, _visit, call_middle=False)

        return model

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        # print(type(forward_res), type(addition_info))
        # exit()
        return forward_res, addition_info
