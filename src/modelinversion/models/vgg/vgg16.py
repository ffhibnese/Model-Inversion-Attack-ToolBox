import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.loss import _Loss

from ..modelresult import ModelResult
from ..evolve import evolve
from torchvision.transforms.functional import resize
from ..base import BaseTargetModel

"""
    FROM PLGMI
"""

class VGG16(BaseTargetModel):
    def __init__(self, n_classes, pretrained=False):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
        
        self.resolution = 64
        
    def get_feature_dim(self):
        return self.feat_dim

    def forward(self, x):
        
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = resize(x, [self.resolution, self.resolution])
            
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return ModelResult(res, [feature])

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out


# class VGG16_vib(nn.Module):
#     def __init__(self, n_classes, pretrained=False):
#         super(VGG16_vib, self).__init__()
#         model = torchvision.models.vgg16_bn(pretrained=pretrained)
#         self.feature = model.features
#         self.feat_dim = 512 * 2 * 2
#         self.k = self.feat_dim // 2
#         self.n_classes = n_classes
#         self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
#         self.fc_layer = nn.Linear(self.k, self.n_classes)
        
#         self.resolution = 64

#     def forward(self, x, mode="train"):
        
#         if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
#             x = resize(x, [self.resolution, self.resolution])
            
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         statis = self.st_layer(feature)
#         mu, std = statis[:, :self.k], statis[:, self.k:]

#         std = F.softplus(std - 5, beta=1)
#         eps = torch.FloatTensor(std.size()).normal_().cuda()
#         res = mu + std * eps
#         out = self.fc_layer(res)

#         output = ModelResult(out, [feature], {'mu': mu, 'std': std})
#         return output

#     def predict(self, x):
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         statis = self.st_layer(feature)
#         mu, std = statis[:, :self.k], statis[:, self.k:]

#         std = F.softplus(std - 5, beta=1)
#         eps = torch.FloatTensor(std.size()).normal_().cuda()
#         res = mu + std * eps
#         out = self.fc_layer(res)

#         return out