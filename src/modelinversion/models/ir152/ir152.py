import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.loss import _Loss
from ...utils import OutputHook
from torchvision.transforms.functional import resize
from ..modelresult import ModelResult
from ..evolve import evolve
from ..base import BaseTargetModel

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class IR152(BaseTargetModel):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        
        self.resolution = 64
        
    def get_feature_dim(self):
        return self.feat_dim
    
    def create_hidden_hooks(self) -> list:
        
        hiddens_hooks = []
        
        length_hidden = len(self.feature.body)
        
        num_body_monitor = 4
        offset = length_hidden // num_body_monitor
        for i in range(num_body_monitor):
            hiddens_hooks.append(OutputHook(self.feature.body[offset * (i+1) - 1]))
        
        hiddens_hooks.append(OutputHook(self.output_layer))
        return hiddens_hooks
    
    def freeze_front_layers(self) -> None:
        length_hidden = len(self.feature.body)
        for i in range(int(length_hidden * 2 // 3)):
            self.feature.body[i].requires_grad_(False)

    def forward(self, x):
        
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = resize(x, [self.resolution, self.resolution])
            
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return ModelResult(out, [feat])


# class IR152_vib(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(IR152_vib, self).__init__()
#         self.feature = evolve.IR_152_64((64, 64))
#         self.feat_dim = 512
#         self.k = self.feat_dim // 2
#         self.n_classes = num_classes
#         self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
#                                           nn.Dropout(),
#                                           Flatten(),
#                                           nn.Linear(512 * 4 * 4, 512),
#                                           nn.BatchNorm1d(512))

#         self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
#         self.fc_layer = nn.Sequential(
#             nn.Linear(self.k, self.n_classes),
#             nn.Softmax(dim=1))
        
#         self.resolution = 64

#     def forward(self, x):
        
#         if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
#             x = resize(x, [self.resolution, self.resolution])
            
#         feature = self.output_layer(self.feature(x))
#         feature = feature.view(feature.size(0), -1)
#         statis = self.st_layer(feature)
#         mu, std = statis[:, :self.k], statis[:, self.k:]

#         std = F.softplus(std - 5, beta=1)
#         eps = torch.FloatTensor(std.size()).normal_().cuda()
#         res = mu + std * eps
#         out = self.fc_layer(res)
#         __, iden = torch.max(out, dim=1)
#         iden = iden.view(-1, 1)
        

#         # return feature, out, iden, mu #, st
#         return ModelResult(out, [feature], {'mu': mu, 'std': std})

