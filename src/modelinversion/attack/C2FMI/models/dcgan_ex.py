import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from itertools import chain
from torchvision import utils
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc0 = nn.Linear(in_features=input_dim, out_features=8*8*1024)

        self.main_module = nn.Sequential(
            # 1024x8x8 -> 512x16x16
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, True),

            # batchx512x16x16 -> batchx256x32x32
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, True),

            # batchx256x32x32 -> batchx128x64x64
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, True),

            # batchx128x64x64 -> batchx64x128x128
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, True),

            # batchx64x128x128 -> batchx3x128x128
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1))

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.fc0(x)
        x = x.view(-1,1024,8,8)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.main_module(x)
        return self.output(x)

class p2img(Generator):
    def __init__(self, num_class):
        super(p2img, self).__init__(4096)
        self.ln = nn.LayerNorm([num_class])
        self.pre_fc = nn.Sequential(
            nn.Linear(num_class, 4096),
            nn.Linear(4096,4096)
        )

    def forward(self, x):
        # x_ = x
        # x = torch.log(x+0.0001)
        x = self.ln(x)
        x = self.pre_fc(x)
        x = self.fc0(x)
        x = x.view(-1,1024,8,8)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.main_module(x)
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main_module = nn.Sequential(
            # 3x128x128 -> 64x64x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # 64x64x64 -> 128x32x32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # 128x32x32 -> 256x16x16
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 256x16x16 -> 512x16x16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 512x16x16 -> 1024x8x8
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(in_features=1024*8*8, out_features=1)

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(-1,1024*8*8)
        out = self.fc(x)
        # out = F.sigmoid(out)
        return out
