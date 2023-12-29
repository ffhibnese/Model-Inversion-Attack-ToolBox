import torch
import torchvision

from torch import nn
from torch.nn import functional as F

from ..base import BaseTargetModel
from ..modelresult import ModelResult
from ...utils.torchutil import OutputHook

class EfficientNet_b0(BaseTargetModel):
    def __init__(self, n_classes, pretrained=False):
        super(EfficientNet_b0, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  ModelResult(res, [feature])
    
    def get_feature_dim(self) -> int:
        return self.feat_dim
    
    def create_hidden_hooks(self) -> list:
        hiddens_hooks = []
        for i, m in enumerate(self.feature[0].children()):
            if i % 2 == 0 and i != 0:
                hiddens_hooks.append(OutputHook(m))
        return hiddens_hooks
                
    def freeze_front_layers(self) -> None:
        freeze_layers = 6
        for i, m in enumerate(self.feature[0].children()):
            if i < freeze_layers:
                for p in m.parameters():
                    p.requires_grad_(False)
            else:
                break

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b1(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(EfficientNet_b1, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  ModelResult(res, [feature])
    
    def get_feature_dim(self) -> int:
        return self.feat_dim
    
    def create_hidden_hooks(self) -> list:
        hiddens_hooks = []
        for i, m in enumerate(self.feature[0].children()):
            if i % 2 == 0 and i != 0:
                hiddens_hooks.append(OutputHook(m))
        return hiddens_hooks
                
    def freeze_front_layers(self) -> None:
        freeze_layers = 6
        for i, m in enumerate(self.feature[0].children()):
            if i < freeze_layers:
                for p in m.parameters():
                    p.requires_grad_(False)
            else:
                break

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b2(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(EfficientNet_b2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1408
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  ModelResult(res, [feature])
    
    def get_feature_dim(self) -> int:
        return self.feat_dim
    
    def create_hidden_hooks(self) -> list:
        hiddens_hooks = []
        for i, m in enumerate(self.feature[0].children()):
            if i % 2 == 0 and i != 0:
                hiddens_hooks.append(OutputHook(m))
        return hiddens_hooks
                
    def freeze_front_layers(self) -> None:
        freeze_layers = 6
        for i, m in enumerate(self.feature[0].children()):
            if i < freeze_layers:
                for p in m.parameters():
                    p.requires_grad_(False)
            else:
                break

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out