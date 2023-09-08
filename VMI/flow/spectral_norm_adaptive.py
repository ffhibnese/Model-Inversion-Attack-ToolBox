
"""
from https://github.com/rtqichen/residual-flows/blob/master/lib/layers/base/lipschitz.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch._six import container_abcs
from itertools import repeat


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

__all__ = ['SpectralNormLinear', 'SpectralNormConv2d', 'LopLinear', 'LopConv2d', 'get_linear', 'get_conv2d']


class SpectralNormLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, n_iterations=None, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 200)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                itrs_used = 0.
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, coeff={}, n_iters={}, atol={}, rtol={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.coeff, self.n_iterations, self.atol,
            self.rtol
        )


class SpectralNormConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, n_iterations=None,
        atol=1e-3, rtol=1e-3, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.initialized = False
        self.register_buffer('spatial_dims', torch.tensor([1., 1.]))
        self.register_buffer('scale', torch.tensor(0.))

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        if self.kernel_size == (1, 1):
            self.register_buffer('u', F.normalize(self.weight.new_empty(self.out_channels).normal_(0, 1), dim=0))
            self.register_buffer('v', F.normalize(self.weight.new_empty(self.in_channels).normal_(0, 1), dim=0))
        else:
            c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
            with torch.no_grad():
                num_input_dim = c * h * w
                v = F.normalize(torch.randn(num_input_dim).to(self.weight), dim=0, eps=1e-12)
                # forward call to infer the shape
                u = F.conv2d(v.view(1, c, h, w), self.weight, stride=self.stride, padding=self.padding, bias=None)
                num_output_dim = u.shape[0] * u.shape[1] * u.shape[2] * u.shape[3]
                self.out_shape = u.shape
                # overwrite u with random init
                u = F.normalize(torch.randn(num_output_dim).to(self.weight), dim=0, eps=1e-12)

                self.register_buffer('u', u)
                self.register_buffer('v', v)

    def compute_weight(self, update=True, n_iterations=None):
        if not self.initialized:
            self._initialize_u_v()
            self.initialized = True

        if self.kernel_size == (1, 1):
            return self._compute_weight_1x1(update, n_iterations)
        else:
            return self._compute_weight_kxk(update, n_iterations)

    def _compute_weight_1x1(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight.view(self.out_channels, self.in_channels)
        if update:
            with torch.no_grad():
                itrs_used = 0
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight.view(self.out_channels, self.in_channels, 1, 1)

    def _compute_weight_kxk(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
        if update:
            with torch.no_grad():
                itrs_used = 0
                for _ in range(n_iterations):
                    old_u = u.clone()
                    old_v = v.clone()
                    v_s = F.conv_transpose2d(
                        u.view(self.out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
                    )
                    v = F.normalize(v_s.view(-1), dim=0, out=v)
                    u_s = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    u = F.normalize(u_s.view(-1), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        weight_v = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff

        # factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        factor =  torch.max(torch.ones(1).to(weight.device).type(sigma.dtype), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        if not self.initialized: self.spatial_dims.copy_(torch.tensor(input.shape[2:4]).to(self.spatial_dims))
        weight = self.compute_weight(update=self.training)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, n_iters={}, atol={}, rtol={}'.format(self.coeff, self.n_iterations, self.atol, self.rtol)
        return s.format(**self.__dict__)
