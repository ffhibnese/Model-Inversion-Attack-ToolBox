import math
import torch
import torch.nn as nn
from .modules import (
    LinearZeros,
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    SqueezeLayer,
    Permute,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from .flow_utils import split_feature, uniform_binning_correction
from .spectral_norm_adaptive import SpectralNormConv2d


def get_block(in_channels, out_channels, hidden_channels, nonlin, sn=False):
    a = {'relu': nn.ReLU, 'elu': nn.ELU}[nonlin]
    if sn:
        block = nn.Sequential(
            SpectralNormConv2d(
                in_channels, hidden_channels, 3, stride=1, padding=1, coeff=1
            ),
            nn.ReLU(inplace=False),
            SpectralNormConv2d(
                hidden_channels, hidden_channels, 1, stride=1, padding=0, coeff=1
            ),
            nn.ReLU(inplace=False),
            SpectralNormConv2d(
                hidden_channels, out_channels, 3, stride=1, padding=1, coeff=1
            ),
        )
    else:
        block = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            a(inplace=False),
            Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            a(inplace=False),
            Conv2dZeros(hidden_channels, out_channels),
        )
    return block


def get_mlp_block(in_channels, out_channels, hidden_channels, nonlin):
    a = {'relu': nn.ReLU, 'elu': nn.ELU}[nonlin]

    block = nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        a(inplace=False),
        nn.Linear(hidden_channels, hidden_channels),
        a(inplace=False),
        LinearZeros(hidden_channels, out_channels),
    )
    return block


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self._forward(input, logdet)
        else:
            output, logdet = self._inverse(input, logdet)
            return output, logdet

    def _forward(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y, None
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=False)

    def _inverse(self, y, logpy=None):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x, None
        return x, logpy + self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=False)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad

    # def __repr__(self):
    #     return ('{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__))


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        use_actnorm,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        sn,
        nonlin,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling
        self.use_actnorm = use_actnorm
        if use_actnorm:
            self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(
                in_channels // 2, in_channels // 2, hidden_channels, nonlin, sn
            )
        elif flow_coupling == "affine":
            self.block = get_block(
                in_channels // 2, in_channels, hidden_channels, nonlin, sn
            )

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        if self.use_actnorm:
            z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        else:
            z = input

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        if self.use_actnorm:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        use_actnorm,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        logittransform,
        sn,
        nonlin,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        if logittransform:
            self.layers.append(LogitTransform(1e-6))
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape
        self.H = H
        self.splits = []

        for i in range(L):
            # 1. Squeeze
            if i != (L - 1) and H != 1:
                C, H, W = C * 4, H // 2, W // 2
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        use_actnorm=use_actnorm,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        sn=sn,
                        nonlin=nonlin,
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                split = Split2d(num_channels=C)
                self.splits.append(split)
                self.layers.append(split)
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        if self.L > 1 and self.H == 1:
            div = 2 ** (self.L - 1)
            C = z.shape[1]
            z = z[:, : C // div]
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class MLPFlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, nonlin):
        super().__init__()
        # ipdb.set_trace()
        self.reverse = Permute(in_channels, shuffle=False)
        self.flow_permutation = lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        self.block = get_mlp_block(
            in_channels // 2, in_channels // 2, hidden_channels, nonlin
        )

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        z = input
        z, logdet = self.flow_permutation(z, logdet, False)

        # # 3. coupling
        z1, z2 = split_feature(z, "split")
        z2 = z2 + self.block(z1)
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = split_feature(input, "split")
        z2 = z2 - self.block(z1)
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        return z, logdet


class MLPFlowNet(nn.Module):
    def __init__(self, hidden_channels, K, nonlin, C=2):
        super().__init__()

        self.layers = nn.ModuleList()

        # 2. K FlowStep
        for _ in range(K):
            self.layers.append(
                MLPFlowStep(
                    in_channels=C, hidden_channels=hidden_channels, nonlin=nonlin
                )
            )

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    def __init__(
        self,
        inp_dim,
        image_shape,
        hidden_channels,
        K,
        L,
        use_actnorm,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        y_classes,
        learn_top,
        y_condition,
        logittransform,
        sn,
        use_binning_correction=True,
        mlp=False,
        nonlin='relu',
    ):
        super().__init__()
        self.mlp = mlp
        if mlp:
            assert L == 0  # L has no effect
            self.flow = MLPFlowNet(
                hidden_channels=hidden_channels, K=K, nonlin=nonlin, C=inp_dim
            )

            self.register_buffer("prior_h", torch.zeros([1, inp_dim * 2]))
        else:
            self.flow = FlowNet(
                image_shape=image_shape,
                hidden_channels=hidden_channels,
                K=K,
                L=L,
                use_actnorm=use_actnorm,
                actnorm_scale=actnorm_scale,
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                LU_decomposed=LU_decomposed,
                logittransform=logittransform,
                sn=sn,
                nonlin=nonlin,
            )

            self.register_buffer(
                "prior_h",
                torch.zeros(
                    [
                        1,
                        self.flow.output_shapes[-1][1] * 2,
                        self.flow.output_shapes[-1][2],
                        self.flow.output_shapes[-1][3],
                    ]
                ),
            )

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.y_classes = y_classes
        self.y_condition = y_condition
        self.use_binning_correction = use_binning_correction
        self.learn_top = learn_top
        self.return_ll_only = False

    def prior(self, data, y_onehot=None, batch_size=0):
        if self.mlp:
            if data is not None:
                h = self.prior_h.repeat(data.shape[0], 1)
            else:
                h = self.prior_h.repeat(batch_size, 1)
        else:
            if data is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
            else:
                # Hardcoded a batch size of 32 here
                h = self.prior_h.repeat(batch_size, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(data.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(
        self,
        x=None,
        y_onehot=None,
        z=None,
        temperature=None,
        reverse=False,
        use_last_split=False,
        batch_size=0,
    ):
        if reverse:
            # ipdb.set_trace()
            assert z is not None or batch_size > 0
            return self.reverse_flow(
                z, y_onehot, temperature, use_last_split, batch_size
            )
        else:
            z, objective, y_logits = self.normal_flow(x, y_onehot)
            if self.return_ll_only:
                return objective
            # Full objective - converted to bits per dimension
            b, c, h, w = x.shape
            bpd = (-objective) / (math.log(2.0) * c * h * w)
            return z, bpd, y_logits

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape
        if self.use_binning_correction:
            x, logdet = uniform_binning_correction(x)
            raise
        else:
            logdet = torch.zeros(b).to(x.device)
        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        return z, objective, y_logits

    def reverse_flow(
        self, z, y_onehot, temperature, use_last_split=False, batch_size=0
    ):
        if z is None:
            mean, logs = self.prior(z, y_onehot, batch_size=batch_size)
            z = gaussian_sample(mean, logs, temperature)
            self._last_z = z.clone()
        if use_last_split:
            for layer in self.flow.splits:
                layer.use_last = True
        x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

    def logp(self, x):
        if self.return_ll_only:
            logp = self(x)
        else:
            logp = self(x)[0]
        return logp

    def sample(self, bs):
        return self(y_onehot=None, temperature=1, batch_size=bs, reverse=True)

    def get_eval_samples(self, bs):
        with torch.no_grad():
            return self(y_onehot=None, temperature=1, batch_size=bs, reverse=True)
