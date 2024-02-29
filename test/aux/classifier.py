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

    # def fit(self,
    #         training_data,
    #         validation_data=None,
    #         test_data=None,
    #         optimizer=None,
    #         lr_scheduler=None,
    #         criterion=nn.CrossEntropyLoss(),
    #         metric=None,
    #         rtpt=None,
    #         config=None,
    #         batch_size=64,
    #         num_epochs=30,
    #         dataloader_num_workers=8,
    #         enable_logging=False,
    #         wandb_init_args=None,
    #         save_base_path="",
    #         config_file=None):

    #     trainloader = DataLoader(training_data,
    #                              batch_size=batch_size,
    #                              shuffle=True,
    #                              num_workers=dataloader_num_workers,
    #                              pin_memory=True)

    #     if rtpt is None:
    #         print('Please use RTPT (Remaining Time to Process Title)')

    #     # Initialize WandB logging
    #     if enable_logging:

    #         if wandb_init_args is None:
    #             wandb_init_args = dict()

    #         wandb_config = {
    #             "Dataset": config.dataset['type'],
    #             'Epochs': num_epochs,
    #             'Batch_size': batch_size,
    #             'Seed': config.seed,
    #             'Initial_lr': optimizer.param_groups[0]['lr'],
    #             'Architecture': self.architecture,
    #             'Pretrained': self.pretrained,
    #             'Optimizer': optimizer,
    #             'Trainingset_size': len(training_data),
    #             'num_classes': self.num_classes,
    #             'Total_parameters':
    #             self.count_parameters(only_trainable=False),
    #             'Trainable_parameters':
    #             self.count_parameters(only_trainable=True)
    #         }

    #         for t in training_data.transform.transforms:
    #             if type(t) is Resize:
    #                 wandb_config['Resize'] = t.size
    #             elif type(t) is RandomCrop:
    #                 wandb_config['RandomCrop'] = t.size
    #             elif type(t) is ColorJitter:
    #                 wandb_config['BrightnessJitter'] = t.brightness
    #                 wandb_config['ContrastJitter'] = t.contrast
    #                 wandb_config['SaturationJitter'] = t.saturation
    #                 wandb_config['HueJitter'] = t.hue
    #             elif type(t) is RandomHorizontalFlip:
    #                 wandb_config['HorizontalFlip'] = t.p

    #         if validation_data:
    #             wandb_config['Validationset_size'] = len(validation_data)

    #         if test_data:
    #             wandb_config['Testset_size'] = len(test_data)

    #         wandb.init(**wandb_init_args, config=wandb_config, reinit=True)
    #         wandb.watch(self.model)
    #         if config_file:
    #             wandb.save(config_file)

    #     # Training cycle
    #     best_model_values = {
    #         'validation_metric': 0.0,
    #         'validation_loss': float('inf'),
    #         'model_state_dict': None,
    #         'model_optimizer_state_dict': None,
    #         'training_metric': 0,
    #         'training_loss': 0,
    #     }

    #     metric_train = metric()

    #     print('----------------------- START TRAINING -----------------------')
    #     for epoch in range(num_epochs):
    #         # Training
    #         print(f'Epoch {epoch + 1}/{num_epochs}')
    #         running_total_loss = 0.0
    #         running_main_loss = 0.0
    #         running_aux_loss = 0.0
    #         metric_train.reset()
    #         self.train()
    #         self.to(self.device)
    #         for inputs, labels in tqdm(trainloader,
    #                                    desc='training',
    #                                    leave=False,
    #                                    file=sys.stdout):
    #             inputs, labels = inputs.to(self.device,
    #                                        non_blocking=True), labels.to(
    #                                            self.device, non_blocking=True)
    #             optimizer.zero_grad()
    #             model_output = self.forward(inputs)
    #             aux_loss = torch.tensor(0.0, device=self.device)

    #             # Separate Inception_v3 outputs
    #             aux_logits = None
    #             if isinstance(model_output, inception.InceptionOutputs):
    #                 if self.model.aux_logits:
    #                     model_output, aux_logits = model_output

    #             main_loss = criterion(model_output, labels)
    #             if aux_logits is not None:
    #                 aux_loss += criterion(aux_logits, labels).sum()

    #             num_samples = inputs.shape[0]
    #             loss = main_loss + aux_loss
    #             loss.backward()
    #             optimizer.step()

    #             running_total_loss += loss * num_samples
    #             running_main_loss += main_loss * num_samples
    #             running_aux_loss += aux_loss * num_samples

    #             metric_train.update(model_output, labels)

    #         print(
    #             f'Training {metric_train.name}:   {metric_train.compute_metric():.2%}',
    #             f'\t Epoch total loss: {running_total_loss / len(training_data):.4f}',
    #             f'\t Epoch main loss: {running_main_loss / len(training_data):.4f}',
    #             f'\t Aux loss: {running_aux_loss / len(training_data):.4f}')

    #         if enable_logging:
    #             wandb.log(
    #                 {
    #                     f'Training {metric_train.name}':
    #                     metric_train.compute_metric(),
    #                     'Training Loss':
    #                     running_total_loss / len(training_data),
    #                     'Learning Rate':
    #                     optimizer.param_groups[0]['lr'],
    #                 },
    #                 step=epoch)

    #         # Validation
    #         if validation_data:
    #             self.eval()
    #             val_metric, val_loss = self.evaluate(
    #                 validation_data,
    #                 batch_size,
    #                 metric,
    #                 criterion,
    #                 dataloader_num_workers=dataloader_num_workers)

    #             print(
    #                 f'Validation {metric_train.name}: {val_metric:.2%} \t Validation Loss:  {val_loss:.4f}'
    #             )

    #             # Save best model
    #             if val_metric > best_model_values['validation_metric']:
    #                 print('Copying better model')
    #                 best_model_values['validation_metric'] = val_metric
    #                 best_model_values['validation_loss'] = val_loss
    #                 best_model_values['model_state_dict'] = deepcopy(
    #                     self.state_dict())
    #                 best_model_values['model_optimizer_state_dict'] = deepcopy(
    #                     optimizer.state_dict())
    #                 best_model_values[
    #                     'training_metric'] = metric_train.compute_metric()
    #                 best_model_values[
    #                     'training_loss'] = running_total_loss / len(
    #                         trainloader)

    #             if enable_logging:
    #                 wandb.log(
    #                     {
    #                         f'Validation {metric_train.name}': val_metric,
    #                         'Validation Loss': val_loss,
    #                     },
    #                     step=epoch)
    #         else:
    #             best_model_values['validation_metric'] = None
    #             best_model_values['validation_loss'] = None
    #             best_model_values['model_state_dict'] = deepcopy(
    #                 self.state_dict())
    #             best_model_values['model_optimizer_state_dict'] = deepcopy(
    #                 optimizer.state_dict())
    #             best_model_values[
    #                 'training_metric'] = metric_train.compute_metric()
    #             best_model_values['training_loss'] = running_total_loss / len(
    #                 trainloader)

    #         # Update the RTPT
    #         if rtpt:
    #             rtpt.step(
    #                 subtitle=f"loss={running_total_loss / len(trainloader):.4f}"
    #             )

    #         # make the lr scheduler step
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()

    #     # save the final model
    #     if validation_data:
    #         self.load_state_dict(best_model_values['model_state_dict'])

    #     if save_base_path:
    #         if not os.path.exists(save_base_path):
    #             os.makedirs(save_base_path)
    #         if validation_data:
    #             model_path = os.path.join(
    #                 save_base_path, self.name +
    #                 f'_{best_model_values["validation_metric"]:.4f}' + '.pth')
    #         else:
    #             model_path = os.path.join(
    #                 save_base_path, self.name +
    #                 f'_{best_model_values["training_metric"]:.4f}_no_val' +
    #                 '.pth')

    #     else:
    #         model_path = self.name

    #     torch.save(
    #         {
    #             'epoch':
    #             num_epochs,
    #             'model_state_dict':
    #             best_model_values['model_state_dict'],
    #             'optimizer_state_dict':
    #             best_model_values['model_optimizer_state_dict'],
    #         }, model_path)

    #     # Test final model
    #     test_metric, test_loss = None, None
    #     if test_data:
    #         test_metric, test_loss = self.evaluate(
    #             test_data,
    #             batch_size,
    #             metric,
    #             criterion,
    #             dataloader_num_workers=dataloader_num_workers)
    #         print(
    #             '----------------------- FINISH TRAINING -----------------------'
    #         )
    #         print(
    #             f'Final Test {metric_train.name}: {test_metric:.2%} \t Test Loss: {test_loss:.4f} \n'
    #         )

    #     if enable_logging:
    #         wandb.save(model_path)
    #         wandb.run.summary[
    #             f'Validation {metric_train.name}'] = best_model_values[
    #                 'validation_metric']
    #         wandb.run.summary['Validation Loss'] = best_model_values[
    #             'validation_loss']
    #         wandb.run.summary[
    #             f'Training {metric_train.name}'] = best_model_values[
    #                 'training_metric']
    #         wandb.run.summary['Training Loss'] = best_model_values[
    #             'training_loss']
    #         wandb.run.summary[f'Test {metric_train.name}'] = test_metric
    #         wandb.run.summary['Test Loss'] = test_loss

    #         wandb.config.update({'model_path': model_path})
    #         wandb.config.update({'config_path': config_file})
    #         wandb.finish()

    # def evaluate(self,
    #              evaluation_data,
    #              batch_size=128,
    #              metric=Accuracy,
    #              criterion=nn.CrossEntropyLoss(),
    #              dataloader_num_workers=4):
    #     evalloader = DataLoader(evaluation_data,
    #                             batch_size=batch_size,
    #                             shuffle=False,
    #                             num_workers=dataloader_num_workers,
    #                             pin_memory=True)
    #     metric = metric()
    #     self.eval()
    #     with torch.no_grad():
    #         running_loss = torch.tensor(0.0, device=self.device)
    #         for inputs, labels in tqdm(evalloader,
    #                                    desc='Evaluating',
    #                                    leave=False,
    #                                    file=sys.stdout):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             model_output = self.forward(inputs)
    #             metric.update(model_output, labels)
    #             running_loss += criterion(model_output,
    #                                       labels).cpu() * inputs.shape[0]

    #         metric_result = metric.compute_metric()

    #         return metric_result, running_loss.item() / len(evaluation_data)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()
