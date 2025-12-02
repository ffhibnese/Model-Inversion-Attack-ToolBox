import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

# from ..arcface.iresnet import *
import sys
from ...utils._iresnet import iresnet50
from .base import *


@register_adapter('p2i_confidence2style')
class P2IConfidence2StyleAdapter(BaseAdapter):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        n_styles=18,
        arcface_model_path=None,
        stride=(1, 1),
        return_out_latent_only=False,
    ):
        super(P2IConfidence2StyleAdapter, self).__init__()

        self.return_out_latent_only = return_out_latent_only

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(arcface_model_path, map_location='cpu'))

        # # input conv layer
        # if video_input:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(
        #             6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        #         ),
        #         *list(resnet50.children())[1:3]
        #     )
        # else:
        self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

        # self.input_layer = nn.Linear(1000, 3*256*256)
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            )

        # --------------CelebA--------------------------------------------------------
        self.l1 = nn.Sequential(
            nn.Linear(num_classes, 64 * 8 * 8 * 8, bias=False),
            nn.BatchNorm1d(64 * 8 * 8 * 8),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            dconv_bn_relu(64 * 8, 64 * 4),
            dconv_bn_relu(64 * 4, 64 * 2),
            dconv_bn_relu(64 * 2, 64),
            dconv_bn_relu(64, 32),
            nn.ConvTranspose2d(32, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
        self.input_dim = num_classes

    def forward(self, x):
        latents = []
        features = []
        # x = x.squeeze(1)
        # print(x.shape, self.input_dim)
        # exit()
        # ------------------------------
        y = self.l1(x)
        y = y.view(y.size(0), -1, 8, 8)
        x = self.l2(y)
        # ------------------------------
        x = x.view(x.size(0), -1, 256, 256)
        outimg = x  # torch.Size([1, 3, 256, 256])
        x = self.conv(x)  # torch.Size([1, 64, 256, 256])
        x = self.block_1(x)  # torch.Size([1, 64, 128, 128])
        features.append(self.avg_pool(x))  # torch.Size([1, 64, 3, 3])
        x = self.block_2(x)  # torch.Size([1, 128, 64, 64])
        features.append(self.avg_pool(x))  # torch.Size([1, 128, 3, 3])
        x = self.block_3(x)  # torch.Size([1, 256, 32, 32])
        content = self.content_layer(x)  # torch.Size([1, 512, 16, 16])
        features.append(self.avg_pool(x))  # torch.Size([1, 256, 3, 3])
        x = self.block_4(x)  # torch.Size([1, 512, 16, 16])
        features.append(self.avg_pool(x))  # torch.Size([1, 512, 3, 3])
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)  # torch.Size([1, 8640])
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)  # torch.Size([1, 18, 512])
        # if self.return_out_latent_only:
        #     return out
        # else:
        return out, content, outimg
        # return out, content, outimg
