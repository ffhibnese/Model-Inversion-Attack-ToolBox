import os
import sys
import pickle
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from .base import BaseIntermediateImageGenerator
from ...utils import check_shape


class StyleGan2adaMappingWrapper(nn.Module):

    def __init__(
        self,
        mapping,
        single_w,
        truncation_psi=0.5,
        truncation_cutoff=8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.single_w = single_w
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.w_dim = mapping.w_dim
        self.z_dim = mapping.z_dim
        self.num_ws = mapping.num_ws

    def forward(self, z):
        if self.mapping:
            w = self.mapping(
                z,
                c=None,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=self.truncation_cutoff,
            )
            if self.single_w:
                w = w[:, [0]]
        else:
            # w = z
            w = z
            # print(w.shape)
            # exit()
        return w


class StyleGAN2adaSynthesisWrapper(BaseIntermediateImageGenerator):

    def __init__(self, synthesis, mapping=None, *args, **kwargs) -> None:
        block_num = len(synthesis.block_resolutions)
        super().__init__(
            synthesis.img_resolution,
            (synthesis.num_ws, synthesis.w_dim),
            block_num,
            *args,
            **kwargs,
        )

        self.synthesis = synthesis
        self.mapping = mapping

    def _forward_impl(
        self,
        ws: Tensor,
        intermediate_inputs: Optional[Tensor] = None,
        labels: torch.LongTensor | None = None,
        start_block: int = None,
        end_block: int = None,
        noise_mode='const',
        force_fp32=True,
        **kwargs,
    ):

        if self.mapping is not None:
            # print('tomap')
            ws = self.mapping(ws)

        if 'noise_mode' not in kwargs:
            kwargs['noise_mode'] = noise_mode
        if 'force_fp32' not in kwargs:
            kwargs['force_fp32'] = force_fp32

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            if ws.shape[-2] == 1:
                ws = torch.repeat_interleave(ws, self.synthesis.num_ws, dim=-2)
            # print(ws.shape)
            # exit()
            check_shape(ws, [None, self.synthesis.num_ws, self.synthesis.w_dim])
            w_idx = 0
            for res in self.synthesis.block_resolutions:
                block = getattr(self.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        x = intermediate_inputs
        for i in range(start_block, end_block):
            res = self.synthesis.block_resolutions[i]
            w = block_ws[i]

            block = getattr(self.synthesis, f'b{res}')
            x, img = block(x, img, w, **kwargs)
        return x if end_block < self.block_num else img


def get_stylegan2ada_generator(
    stylegan2ada_path: str,
    checkpoint_path: str,
    single_w=True,
    truncation_psi=0.5,
    truncation_cutoff=8,
    optimize_in_w=True,
):

    sys.path.append(stylegan2ada_path)

    with open(checkpoint_path, 'rb') as f:
        G = pickle.load(f)['G_ema']

    _mapping = StyleGan2adaMappingWrapper(
        G.mapping, single_w, truncation_psi, truncation_cutoff
    )
    if optimize_in_w:
        mapping = _mapping
        synthesis = StyleGAN2adaSynthesisWrapper(G.synthesis)
    else:
        mapping = nn.Identity()
        mapping.single_w = single_w
        mapping.truncation_psi = truncation_psi
        mapping.truncation_cutoff = truncation_cutoff
        mapping.w_dim = G.mapping.w_dim
        mapping.z_dim = G.mapping.z_dim
        mapping.num_ws = G.mapping.num_ws

        synthesis = StyleGAN2adaSynthesisWrapper(G.synthesis, _mapping)

    return mapping, synthesis


def get_stylegan2ada_discriminator(
    stylegan2ada_path: str,
    checkpoint_path: str,
    # single_w=True,
    # truncation_psi=0.5,
    # truncation_cutoff=8,
    # optimize_in_w=True,
):

    sys.path.append(stylegan2ada_path)

    with open(checkpoint_path, 'rb') as f:
        D = pickle.load(f)['D']

    return D
