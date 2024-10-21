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
from .wrappers import *
from .evolve.evolve import bottleneck_IR, bottleneck_IR_SE


from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.densenet import DenseNet, _DenseBlock
from torchvision.models.efficientnet import MBConv, FusedMBConv
from torchvision.models.vision_transformer import EncoderBlock as ViTEncoderBlock
from torchvision.models.swin_transformer import (
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)


def default_erase_residule(module: nn.Module, residule_keep_ratio: float = 0.0):

    origin_forward = module.forward
    add_ratio = -1 + residule_keep_ratio

    def forward(self, *args, _add_ratio: int = add_ratio, **kwargs):
        identity = args[0]
        out = origin_forward(*args, **kwargs)
        if isinstance(out, tuple):
            out[0] += identity * _add_ratio
        else:
            print(out.shape, identity.shape, _add_ratio)
            # exit()
            out += identity * _add_ratio
        return out

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_ir_block(
    module: bottleneck_IR | bottleneck_IR_SE, residule_keep_ratio: float = 0.0
):
    if not isinstance(module, (bottleneck_IR, bottleneck_IR_SE)):
        raise ValueError('The module is not bottleneck_IR or bottleneck_IR_SE')

    def forward(
        self, x: Tensor, _residule_keep_ratio: int = residule_keep_ratio
    ) -> Tensor:
        shortcut = self.shortcut_layer(x)

        res = self.res_layer(x)

        return res + shortcut * _residule_keep_ratio

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_denseblock(module: _DenseBlock, residule_keep_ratio: float = 0.0):
    if not isinstance(module, _DenseBlock):
        raise ValueError('The module is not _DenseBlock')

    def forward(
        self, init_features: Tensor, _residule_keep_ratio: int = residule_keep_ratio
    ) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            channels = features[-1].shape[1]
            channel_idx = round(channels * _residule_keep_ratio)
            new_features[:, channel_idx:] = 0
            features.append(new_features)
        return torch.cat(features, 1)

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_resnet_basicblock(
    module: BasicBlock, residule_keep_ratio: float = 0.0
):
    if not isinstance(module, BasicBlock):
        raise ValueError('The module is not BasicBlock')

    def forward(
        self, x: Tensor, _residule_keep_ratio: int = residule_keep_ratio
    ) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity * _residule_keep_ratio
        out = self.relu(out)

        return out

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_resnet_bottleneck(
    module: Bottleneck, residule_keep_ratio: float = 0.0
):
    if not isinstance(module, Bottleneck):
        raise ValueError('The module is not Bottleneck')

    def forward(
        self, x: Tensor, _residule_keep_ratio: int = residule_keep_ratio
    ) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity * _residule_keep_ratio
        out = self.relu(out)

        return out

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_mbconv(
    module: MBConv | FusedMBConv, residule_keep_ratio: float = 0.0
):
    if not isinstance(module, MBConv | FusedMBConv):
        raise ValueError('The module is not MBConv or FusedMBConv')

    if not module.use_res_connect:
        return False

    def forward(
        self, input: Tensor, _residule_keep_ratio: int = residule_keep_ratio
    ) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input * _residule_keep_ratio
        return result

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_vit_encoderblock(
    module: ViTEncoderBlock, residule_keep_ratio: float = 0.0
):
    if not isinstance(module, ViTEncoderBlock):
        raise ValueError('The module is not EncoderBlock')

    def forward(
        self, input: torch.Tensor, _residule_keep_ratio: float = residule_keep_ratio
    ):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x * _residule_keep_ratio + y

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_swin_block(
    module: SwinTransformerBlock, residule_keep_ratio: float = 0.0
):

    if not isinstance(module, SwinTransformerBlock):
        raise ValueError('The module is not SwinTransformerBlock')

    def forward(
        self, x: torch.Tensor, _residule_keep_ratio: float = residule_keep_ratio
    ):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x * _residule_keep_ratio + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x

    module.forward = forward.__get__(module, module.__class__)

    return True


def erase_residule_swin_v2_block(
    module: SwinTransformerBlock, residule_keep_ratio: float = 0.0
):

    if not isinstance(module, SwinTransformerBlockV2):
        raise ValueError('The module is not SwinTransformerBlockV2')

    def forward(self, x: Tensor, _residule_keep_ratio: float = residule_keep_ratio):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x * _residule_keep_ratio + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x

    module.forward = forward.__get__(module, module.__class__)

    return True


# def get_remove_residule_forward_hook(residule_keep_ratio: int = 0):

#     add_coef = -1 + residule_keep_ratio

#     def remove_residule_forward_hook(module, input, output):
#         if isinstance(module, torch.Tensor):
#             output -= input * add_coef
#         else:
#             output[0] -= input * add_coef
#         return output

#     return remove_residule_forward_hook

REMOVE_FUNCTIONS = {
    BasicBlock: erase_residule_resnet_basicblock,
    Bottleneck: erase_residule_resnet_bottleneck,
    MBConv: erase_residule_mbconv,
    FusedMBConv: erase_residule_mbconv,
    bottleneck_IR: erase_residule_ir_block,
    bottleneck_IR_SE: erase_residule_ir_block,
    ViTEncoderBlock: erase_residule_vit_encoderblock,
    SwinTransformerBlock: erase_residule_swin_block,
    SwinTransformerBlockV2: erase_residule_swin_v2_block,
    _DenseBlock: erase_residule_denseblock,
}


def add_remove_functions(type_module, func):
    REMOVE_FUNCTIONS[type_module] = func


def remove_last_residule(
    module, residule_keep_ratio: int = 0, erase_ratio_or_num: float | int = 0
):

    erase_fns = []

    def _visit_fn(m):
        # modules.append(m)
        type_m = m.__class__
        # if m.__class__.__name__ == 'bottleneck_IR':
        # print(m.__class__)
        if type_m in REMOVE_FUNCTIONS:
            erase_fns.append((REMOVE_FUNCTIONS[type_m], m))

    traverse_module(module, _visit_fn, call_middle=True)

    erase_num = (
        len(erase_fns) * erase_ratio_or_num
        if isinstance(erase_ratio_or_num, float)
        else erase_ratio_or_num
    )

    for i, (fn, m) in enumerate(erase_fns[::-1]):
        # if fn(m, residule_keep_ratio):
        fn(m, residule_keep_ratio)
        if i >= erase_num:
            return

    raise ValueError('The module is not support for skip connection')

    # return remove_module.register_forward_hook(
    #     get_remove_residule_forward_hook(residule_keep_ratio)
    # )


@register_model('skipconeection')
class SkipConnectionWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        residule_keep_ratio: int = 0,
        erase_ratio_or_num: float | int = 0,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.skip_hook = remove_last_residule(
            module, residule_keep_ratio, erase_ratio_or_num
        )

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        return self.module(image, *args, **kwargs)
