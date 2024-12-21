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


class _ConditionMLP1d(nn.Module):

    def __init__(
        self,
        in_features,
        cond_features_num,
        cond_features_dim,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features + cond_features_dim, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

        self.cond_embedding = nn.Embedding(cond_features_num, cond_features_dim)

    def forward(self, x, cond):
        ori_x = x
        cond = self.cond_embedding(cond)
        x = torch.cat([x, cond], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x) + ori_x
        # x = self.drop(x)
        return x


@register_model('cond_purifier')
class ConditionPurifierWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        cond_features_dim=512,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.module.eval()

        for p in self.module.parameters():
            p.requires_grad = False

        self.purifier = _ConditionMLP1d(
            in_features=self.module.num_classes,
            cond_features_num=self.module.num_classes,
            cond_features_dim=cond_features_dim,
            hidden_features=cond_features_dim // 2,
            out_features=self.module.feature_dim,
            act_layer=nn.ReLU,
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.purifier.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        return self.purifier.train(mode)

    def eval(self):
        return self.purifier.eval()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        addition_info['ori_logits'] = forward_res
        cond = torch.argmax(forward_res, dim=-1)
        forward_res = self.purifier(forward_res, cond)
        return forward_res, addition_info


class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation)
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x


@register_model('adv_purifier')
class AdversarialPurifierWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        path: str,
        device: str,
        iteration: int,
        eta: float,
        eps: float,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.module.eval()
        self.iteration = iteration
        self.eta = eta
        self.eps = eps

        for p in self.module.parameters():
            p.requires_grad = False

        self.inversion = Inversion(nc=1, ngf=128, nz=530, truncation=530, c=50)
        self.inversion.load_state_dict(
            state_dict=torch.load(path, map_location='cpu'), strict=True
        )
        self.inversion.to(device=device)
        self.inversion.eval()

    def get_noise(
        self, ori_logit: Tensor, cur_logit: Tensor, grad: Tensor, eta: float, eps: float
    ):
        logit = cur_logit + eta * torch.sign(grad)
        l = torch.argmax(ori_logit)
        m = torch.relu(torch.max(logit) - logit[l])
        logit[l] += m
        noise = logit - ori_logit
        if torch.norm(noise, p=1) > eps:
            noise *= eps / torch.norm(noise, p=1)
        return noise

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.purifier.inversion(recurse=recurse)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        addition_info['ori_logits'] = forward_res
        ori_logit = forward_res
        cur_logit = torch.tensor(forward_res, requires_grad=True)
        optimizer = torch.optim.Adam(
            cur_logit, lr=0.0002, betas=(0.5, 0.999), amsgrad=True
        )
        for i in range(self.iteration):
            recon = self.inversion(torch.softmax(cur_logit))
            loss = F.mse_loss(recon, image)
            loss.backward()
            cur_logit = ori_logit + self.get_noise(
                ori_logit=ori_logit,
                cur_logit=cur_logit,
                grad=loss.grad,
                eta=self.eta,
                eps=self.eps,
            )
        return torch.softmax(cur_logit), addition_info
