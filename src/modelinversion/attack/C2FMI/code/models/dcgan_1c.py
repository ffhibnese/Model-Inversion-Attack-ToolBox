import torch
import torch.nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, num_class, trunc):
        super(Generator, self).__init__()

        self.trunc = trunc
        self.num_class = num_class
        self.main_module = nn.Sequential(
            # 526x1x1 -> 1024x4x4
            nn.ConvTranspose2d(
                in_channels=num_class,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.Tanh(),
            # batchx1024x4x4 -> batchx512x8x8
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.Tanh(),
            # batchx512x8x8 -> batchx256x16x16
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.Tanh(),
            # batchx256x16x16 -> batchx128x32x32
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.Tanh(),
            # batchx128x32x32 -> batchx1x64x64
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)

        x = x.view(-1, self.num_class, 1, 1)
        x = self.main_module(x)
        x = x.view(-1, 1, 64, 64)
        return x
