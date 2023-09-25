# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 5, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return [feature, out]

class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        self.feat_dim = 512
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return [out]


