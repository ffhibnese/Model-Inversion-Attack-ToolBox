from copy import deepcopy

from torch import Tensor
import torchvision

from ..utils import ModelMixin
from ...utils import BaseHook

from .base import *
from .evolve import evolve


@register_model('vgg16_64')
class VGG16_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes, pretrained=False, register_last_feature_hook=False):
        self.feat_dim = 512 * 2 * 2
        super(VGG16_64, self).__init__(
            64, self.feat_dim, num_classes, register_last_feature_hook
        )
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.feature = model.features

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.feature_hook = FirstInputHook(self.fc_layer)

    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook

    # def create_hidden_hooks(self) -> list:

    #     hiddens_hooks = []
    #     def _add_hook_fn(module):
    #         if isinstance(module, MaxPool2d):
    #             hiddens_hooks.append(OutputHook(module))
    #     traverse_module(self, _add_hook_fn, call_middle=False)
    #     return hiddens_hooks

    # def freeze_front_layers(self) -> None:

    #     freeze_num = 8
    #     i = 0
    #     for m in self.feature.children():

    #         if isinstance(m, nn.Conv2d):
    #             i += 1
    #             if i >= freeze_num:
    #                 break
    #         for p in m.parameters():
    #             p.requires_grad_(False)

    def _forward_impl(self, x: Tensor, *args, **kwargs):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return res


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)


@register_model(name='ir152_64')
class IR152_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        register_last_feature_hook=False,
        backbone_path: Optional[str] = None,
    ):
        self.feat_dim = 512
        super(IR152_64, self).__init__(
            64, self.feat_dim, num_classes, register_last_feature_hook
        )
        self.feature = evolve.IR_152_64((64, 64))
        if backbone_path is not None:
            state_dict = torch.load(backbone_path, map_location='cpu')
            for k in list(state_dict.keys()):
                if 'output_layer' in k:
                    del state_dict[k]
            self.feature.load_state_dict(state_dict)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.feature_hook = FirstInputHook(self.fc_layer)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['backbone_path']
        return super().preprocess_config_before_save(config)

    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook

    # def create_hidden_hooks(self) -> list:

    #     hiddens_hooks = []

    #     length_hidden = len(self.feature.body)

    #     num_body_monitor = 4
    #     offset = length_hidden // num_body_monitor
    #     for i in range(num_body_monitor):
    #         hiddens_hooks.append(OutputHook(self.feature.body[offset * (i+1) - 1]))

    #     hiddens_hooks.append(OutputHook(self.output_layer))
    #     return hiddens_hooks

    # def freeze_front_layers(self) -> None:
    #     length_hidden = len(self.feature.body)
    #     for i in range(int(length_hidden * 2 // 3)):
    #         self.feature.body[i].requires_grad_(False)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out


@register_model(name='facenet64')
class FaceNet64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        register_last_feature_hook=False,
        backbone_path: Optional[str] = None,
    ):
        super(FaceNet64, self).__init__(
            64, 512, num_classes, register_last_feature_hook
        )
        self.feature = evolve.IR_50_64((64, 64))
        if backbone_path is not None:
            state_dict = torch.load(backbone_path, map_location='cpu')
            for k in list(state_dict.keys()):
                if 'output_layer' in k:
                    del state_dict[k]
            self.feature.load_state_dict(state_dict)
        self.feat_dim = 512
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.feature_hook = FirstInputHook(self.fc_layer)

    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['backbone_path']
        return super().preprocess_config_before_save(config)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out


@register_model(name='efficientnet_b0_64')
class EfficientNet_b0_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1000, prtrained=False):
        super(EfficientNet_b0_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=prtrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim


@register_model(name='efficientnet_b1_64')
class EfficientNet_b1_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1000, prtrained=False):
        super(EfficientNet_b1_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=prtrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim


@register_model(name='efficientnet_b2_64')
class EfficientNet_b2_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1408, prtrained=False):
        super(EfficientNet_b2_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=prtrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1408
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim
