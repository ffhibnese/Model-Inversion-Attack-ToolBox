import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d
import torchvision.models
from torchvision.transforms.functional import resize

from ...utils import OutputHook, traverse_module
from ..modelresult import ModelResult
from .. import BaseTargetModel


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

    def create_hidden_hooks(self) -> list:

        hiddens_hooks = []

        def _add_hook_fn(module):
            if isinstance(module, MaxPool2d):
                hiddens_hooks.append(OutputHook(module))

        traverse_module(self, _add_hook_fn, call_middle=False)
        return hiddens_hooks

    def freeze_front_layers(self) -> None:

        freeze_num = 8
        i = 0
        for m in self.feature.children():

            if isinstance(m, nn.Conv2d):
                i += 1
                if i >= freeze_num:
                    break
            for p in m.parameters():
                p.requires_grad_(False)

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
