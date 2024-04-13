import math
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base import BaseIntermediateImageGenerator, LambdaModule


class _ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(_ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class _CategoricalConditionalBatchNorm2d(_ConditionalBatchNorm2d):

    def __init__(
        self,
        num_classes,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(_CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(_CategoricalConditionalBatchNorm2d, self).forward(
            input, weight, bias
        )


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class _GenBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        upsample=False,
        num_classes=0,
    ):
        super(_GenBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = _CategoricalConditionalBatchNorm2d(num_classes, in_ch)
            self.b2 = _CategoricalConditionalBatchNorm2d(num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class PlgmiGenerator64(BaseIntermediateImageGenerator):
    """Generator generates 64x64."""

    def __init__(self, num_classes, dim_z=128, bottom_width=4):
        super(PlgmiGenerator64, self).__init__(64, dim_z, 6)
        activation = nn.ReLU()
        self.num_features = num_features = 64
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        # self.distribution = distribution

        def _reshape():
            return LambdaModule(
                lambda x: x.reshape(x.size(0), -1, self.bottom_width, self.bottom_width)
            )

        # print(dim_z)
        self.block1 = nn.Sequential(
            nn.Linear(dim_z, 16 * num_features * bottom_width**2), _reshape()
        )

        self.block2 = _GenBlock(
            num_features * 16,
            num_features * 8,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = _GenBlock(
            num_features * 8,
            num_features * 4,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = _GenBlock(
            num_features * 4,
            num_features * 2,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block5 = _GenBlock(
            num_features * 2,
            num_features,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block5_res = nn.Sequential(nn.BatchNorm2d(num_features), activation)

        self.block6 = nn.Sequential(nn.Conv2d(num_features, 3, 1, 1), nn.Tanh())

    def _forward_impl(
        self,
        *inputs,
        labels: torch.LongTensor | None = None,
        start_block: int = None,
        end_block: int = None,
        **kwargs,
    ):
        h = inputs[0]

        for i in range(start_block, end_block):
            if i in [0, 5]:
                h = getattr(self, f'block{i+1}')(h)
            else:
                h = getattr(self, f'block{i+1}')(h, labels, **kwargs)
            if i == 4:
                h = self.block5_res(h)
        return h


class PlgmiGenerator256(BaseIntermediateImageGenerator):
    """Generator generates 64x64."""

    def __init__(self, num_classes, dim_z=128, bottom_width=4):
        super(PlgmiGenerator256, self).__init__(256, dim_z, 8)
        activation = nn.ReLU()
        self.num_features = num_features = 64
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        # self.distribution = distribution

        def _reshape():
            return LambdaModule(
                lambda x: x.reshape(x.size(0), -1, self.bottom_width, self.bottom_width)
            )

        # print(dim_z)
        self.block1 = nn.Sequential(
            nn.Linear(dim_z, 16 * num_features * bottom_width**2), _reshape()
        )

        self.block2 = _GenBlock(
            num_features * 16,
            num_features * 8,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = _GenBlock(
            num_features * 8,
            num_features * 4,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = _GenBlock(
            num_features * 4,
            num_features * 2,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block5 = _GenBlock(
            num_features * 2,
            num_features,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block6 = _GenBlock(
            num_features * 2,
            num_features,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block7 = _GenBlock(
            num_features * 2,
            num_features,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block7_res = nn.Sequential(nn.BatchNorm2d(num_features), activation)

        self.block8 = nn.Sequential(nn.Conv2d(num_features, 3, 1, 1), nn.Tanh())

    def _forward_impl(
        self,
        *inputs,
        labels: torch.LongTensor | None = None,
        start_block: int = None,
        end_block: int = None,
        **kwargs,
    ):
        h = inputs[0]
        for i in range(start_block, end_block):
            if i in [0, 7]:
                h = getattr(self, f'block{i+1}')(h)
            else:
                h = getattr(self, f'block{i+1}')(h, labels, **kwargs)
            if i == 6:
                h = self.block7_res(h)
        return h


class _DisBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        downsample=False,
    ):
        super(_DisBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class _OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(_OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class PlgmiDiscriminator64(nn.Module):

    def __init__(self, num_classes):
        super(PlgmiDiscriminator64, self).__init__()

        num_features = 64
        activation = F.relu
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = _OptimizedBlock(3, num_features)
        self.block2 = _DisBlock(
            num_features, num_features * 2, activation=activation, downsample=True
        )
        self.block3 = _DisBlock(
            num_features * 2, num_features * 4, activation=activation, downsample=True
        )
        self.block4 = _DisBlock(
            num_features * 4, num_features * 8, activation=activation, downsample=True
        )
        self.block5 = _DisBlock(
            num_features * 8, num_features * 16, activation=activation, downsample=True
        )
        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        # if num_classes > 0:
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        # if y is not None:
        output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class PlgmiDiscriminator256(nn.Module):

    def __init__(self, num_classes):
        super(PlgmiDiscriminator256, self).__init__()

        num_features = 64
        activation = F.relu
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = _OptimizedBlock(3, num_features)
        self.block2 = _DisBlock(
            num_features, num_features * 2, activation=activation, downsample=True
        )
        self.block3 = _DisBlock(
            num_features * 2, num_features * 4, activation=activation, downsample=True
        )
        self.block4 = _DisBlock(
            num_features * 4, num_features * 8, activation=activation, downsample=True
        )
        self.block5 = _DisBlock(
            num_features * 8, num_features * 16, activation=activation, downsample=True
        )
        self.block6 = _DisBlock(
            num_features * 16, num_features * 16, activation=activation, downsample=True
        )
        self.block7 = _DisBlock(
            num_features * 16, num_features * 16, activation=activation, downsample=True
        )
        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        # if num_classes > 0:
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        # if y is not None:
        output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


LoktGenerator64 = PlgmiGenerator64
LoktGenerator256 = PlgmiGenerator256


class LoktDiscriminator64(nn.Module):

    def __init__(self, num_classes):
        super(LoktDiscriminator64, self).__init__()

        num_features = 64
        activation = F.relu
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = _OptimizedBlock(3, num_features)
        self.block2 = _DisBlock(
            num_features, num_features * 2, activation=activation, downsample=True
        )
        self.block3 = _DisBlock(
            num_features * 2, num_features * 4, activation=activation, downsample=True
        )
        self.block4 = _DisBlock(
            num_features * 4, num_features * 8, activation=activation, downsample=True
        )
        self.block5 = _DisBlock(
            num_features * 8, num_features * 16, activation=activation, downsample=True
        )
        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        # if num_classes > 0:
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = torch.sigmoid(self.l6(h))
        # if y is not None:
        pred = self.l_y(h)
        return output, pred


class LoktDiscriminator256(nn.Module):

    def __init__(self, num_classes):
        super(LoktDiscriminator256, self).__init__()

        num_features = 64
        activation = F.relu
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = _OptimizedBlock(3, num_features)
        self.block2 = _DisBlock(
            num_features, num_features * 2, activation=activation, downsample=True
        )
        self.block3 = _DisBlock(
            num_features * 2, num_features * 4, activation=activation, downsample=True
        )
        self.block4 = _DisBlock(
            num_features * 4, num_features * 8, activation=activation, downsample=True
        )
        self.block5 = _DisBlock(
            num_features * 8, num_features * 16, activation=activation, downsample=True
        )
        self.block6 = _DisBlock(
            num_features * 16, num_features * 16, activation=activation, downsample=True
        )
        self.block7 = _DisBlock(
            num_features * 16, num_features * 16, activation=activation, downsample=True
        )
        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        # if num_classes > 0:
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = torch.sigmoid(self.l6(h))
        # if y is not None:
        pred = self.l_y(h)
        return output, pred
