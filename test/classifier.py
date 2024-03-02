import math
import os
import sys
from copy import deepcopy

import numpy as np
import timm
import torch
import torch.nn as nn
import wandb
# from metrics.accuracy import Accuracy
from torch.utils.data import DataLoader
from torchvision.models import densenet, inception, resnet
from torchvision.transforms import (ColorJitter, RandomCrop,
                                    RandomHorizontalFlip, Resize)
from tqdm import tqdm

from base_model import BaseModel

import sys
from ResNeSt.resnest.torch import resnest50, resnest101, resnest200, resnest269

class Classifier(BaseModel):

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 architecture='resnet18',
                 pretrained=False,
                 name='Classifier',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.model = self._build_model(architecture, pretrained)
        self.model.to(self.device)
        self.architecture = architecture

        self.to(self.device)

    def _build_model(self, architecture, pretrained):
        architecture = architecture.lower().replace('-',
                                                    '').replace('_',
                                                                '').strip()
        if 'resnet' in architecture:
            if architecture == 'resnet18':
                weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
                model = resnet.resnet18(weights=weights)
            elif architecture == 'resnet34':
                weights = resnet.ResNet34_Weights.DEFAULT if pretrained else None
                model = resnet.resnet34(weights=weights)
            elif architecture == 'resnet50':
                weights = resnet.ResNet50_Weights.DEFAULT if pretrained else None
                model = resnet.resnet50(weights=weights)
            elif architecture == 'resnet101':
                weights = resnet.ResNet101_Weights.DEFAULT if pretrained else None
                model = resnet.resnet101(weights=weights)
            elif architecture == 'resnet152':
                weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                model = resnet.resnet152(weights=weights)
            else:
                raise RuntimeError(
                    f'No RationalResNet with the name {architecture} available'
                )

            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            return model

        elif 'resnext' in architecture:
            if architecture == 'resnext50':
                weights = resnet.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
                model = resnet.resnext50_32x4d(weights=weights)
            elif architecture == 'resnext101':
                weights = resnet.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
                model = resnet.resnext101_32x8d(weights=weights)
            else:
                raise RuntimeError(
                    f'No ResNext with the name {architecture} available')

            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            return model

        elif 'resnest' in architecture:
            # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
            print(f'start load {architecture}')
            if architecture == 'resnest50':
                # model = torch.hub.load('zhanghang1989/ResNeSt',
                #                        'resnest50',
                #                        pretrained=True)
                model = resnest50()
            elif architecture == 'resnest101':
                # model = torch.hub.load('zhanghang1989/ResNeSt',
                #                        'resnest101',
                #                        pretrained=True)
                model = resnest101()
            elif architecture == 'resnest200':
                # model = torch.hub.load('zhanghang1989/ResNeSt',
                #                        'resnest200',
                #                        pretrained=True)
                model = resnest200()
            elif architecture == 'resnest269':
                # model = torch.hub.load('zhanghang1989/ResNeSt',
                #                        'resnest269',
                #                        pretrained=True)
                model = resnest269()
            else:
                raise RuntimeError(
                    f'No ResNeSt with the name {architecture} available')

            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            print(f'end load {architecture}')
            return model

        elif 'densenet' in architecture:
            if architecture == 'densenet121':
                weights = densenet.DenseNet121_Weights.DEFAULT if pretrained else None
                model = densenet.densenet121(weights=weights)
            elif architecture == 'densenet161':
                weights = densenet.DenseNet161_Weights.DEFAULT if pretrained else None
                model = densenet.densenet161(weights=weights)
            elif architecture == 'densenet169':
                weights = densenet.DenseNet169_Weights.DEFAULT if pretrained else None
                model = densenet.densenet169(weights=weights)
            elif architecture == 'densenet201':
                weights = densenet.DenseNet201_Weights.DEFAULT if pretrained else None
                model = densenet.densenet201(weights=weights)
            else:
                raise RuntimeError(
                    f'No DenseNet with the name {architecture} available')

            if self.num_classes != model.classifier.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.classifier = nn.Linear(model.classifier.in_features,
                                             self.num_classes)
            return model

        # Note: inception_v3 expects input tensors with a size of N x 3 x 299 x 299, aux_logits are used per default
        elif 'inception' in architecture:
            weights = inception.Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception.inception_v3(weights=weights,
                                           aux_logits=True,
                                           init_weights=True)
            if self.num_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        elif 'vit' in architecture:
            if architecture == 'vitb16':
                model = timm.create_model('vit_base_patch16_224',
                                          pretrained=pretrained)
            elif architecture == 'vitb32':
                model = timm.create_model('vit_base_patch32_224',
                                          pretrained=pretrained)
            elif architecture == 'vitl16':
                model = timm.create_model('vit_large_patch16_224',
                                          pretrained=pretrained)
            elif architecture == 'vitl32':
                model = timm.create_model('vit_large_patch32_224',
                                          pretrained=pretrained)
            elif architecture == 'vith14':
                model = timm.create_model('vit_huge_patch14_224',
                                          pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No ViT with the name {architecture} available')

            if self.num_classes != model.head.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.head = nn.Linear(model.head.in_features,
                                       self.num_classes)
            return model

        else:
            raise RuntimeError(
                f'No network with the name {architecture} available')

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        out = self.model(x)
        return out


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()


if __name__ == '__main__':
    
    import sys
    sys.path.append('..')
    sys.path.append('../src')
    from modelinversion.models.defense_wrapper import TorchVisionModelWrapper
    
    model_name = 'resnet152'
    # /data/yuhongyao/intermediate-MIA/intermediate-MIA/pretrained/resnet152_facescrub_negative_LS.pt
    model = Classifier(1000, architecture=model_name)
    state_dict = torch.load(f'/data/yuhongyao/intermediate-MIA/intermediate-MIA/pretrained/{model_name}_celeba.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    
    ourmodel = TorchVisionModelWrapper(model.model, 1000)
    ourmodel.model.load_state_dict(model.model.state_dict())
    save_path = f'/data/yuhongyao/Model_Inversion_Attack_ToolBox/checkpoints/target_eval/hdceleba/{model_name}_celeba.pt'
    torch.save({'state_dict': ourmodel.state_dict()}, save_path)