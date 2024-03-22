
from typing import Union, Optional

import torch
import torch.nn as nn
from .base import BaseIntermediateImageGenerator, LambdaModule

class SimpleGenerator64(BaseIntermediateImageGenerator):
    def __init__(self, in_dim=100):
        super(SimpleGenerator64, self).__init__(64, in_dim, 5)
        
        dim=64
        
        def _reshape4x4():
            return LambdaModule(lambda x: x.view(len(x), -1, 4, 4))

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
            _reshape4x4())
        self.l2_5 = nn.Sequential(
            l1,
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.Sequential(
                nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
                nn.Sigmoid()
            )
        )
        
        
        
    def _forward_impl(self, *inputs, labels: torch.LongTensor | None = None, start_block: int = None, end_block: int = None, **kwargs):
        x = inputs[0]
        blocks = self.l2_5[start_block:end_block]
        return blocks[x]

class SimpleGenerator256(BaseIntermediateImageGenerator):
    def __init__(self, in_dim=100):
        
        super(SimpleGenerator64, self).__init__(256, in_dim, 7)
        
        dim=64
        
        def _reshape4x4():
            return LambdaModule(lambda x: x.view(len(x), -1, 4, 4))

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
            _reshape4x4())
        self.l2_5 = nn.Sequential(
            l1,
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            dconv_bn_relu(dim, dim),
            dconv_bn_relu(dim, dim),
            nn.Sequential(
                nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
                nn.Sigmoid()
            )
        )
        
        
        
    def _forward_impl(self, *inputs, labels: torch.LongTensor | None = None, start_block: int = None, end_block: int = None, **kwargs):
        x = inputs[0]
        blocks = self.l2_5[start_block:end_block]
        return blocks[x]
    
class DGWGAN64(nn.Module):
    def __init__(self):
        super(DGWGAN64, self).__init__()
        
        in_dim=3
        dim=64

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
    
class DGWGAN256(nn.Module):
    def __init__(self):
        super(DGWGAN64, self).__init__()
        
        in_dim=3
        dim=64

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            conv_ln_lrelu(dim * 8, dim * 8),
            conv_ln_lrelu(dim * 8, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
    
class _MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x
    
class MinibatchDiscriminator64(nn.Module):
    def __init__(self, num_classes):
        super(MinibatchDiscriminator64, self).__init__()
        
        in_dim=3
        dim=64
        self.n_classes = num_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.mbd1 = _MinibatchDiscrimination(dim * 4 * 4 * 4, 64, 50)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4 + 64, self.n_classes)

    def forward(self, x):
        # out = []
        bs = x.shape[0]
        feat1 = self.layer1(x)
        # out.append(feat1)
        feat2 = self.layer2(feat1)
        # out.append(feat2)
        feat3 = self.layer3(feat2)
        # out.append(feat3)
        feat4 = self.layer4(feat3)
        # out.append(feat4)
        feat = feat4.view(bs, -1)
        # print('feat:', feat.shape)
        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y
    
class MinibatchDiscriminator256(nn.Module):
    def __init__(self, num_classes):
        super(MinibatchDiscriminator256, self).__init__()
        
        in_dim=3
        dim=64
        self.n_classes = num_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer5 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer6 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.mbd1 = _MinibatchDiscrimination(dim * 4 * 4 * 4, 64, 50)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4 + 64, self.n_classes)

    def forward(self, x):
        # out = []
        bs = x.shape[0]
        feat1 = self.layer1(x)
        # out.append(feat1)
        feat2 = self.layer2(feat1)
        # out.append(feat2)
        feat3 = self.layer3(feat2)
        # out.append(feat3)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)
        feat6 = self.layer6(feat5)
        # out.append(feat4)
        feat = feat6.view(bs, -1)
        # print('feat:', feat.shape)
        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y