import sys
import os
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    Resize,
)

from modelinversion.models import IR152_64, LoraWrapper, NeckWrapper
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss
from modelinversion.datasets import FaceScrub

if __name__ == '__main__':

    num_classes = 530
    model_name = 'ir152'
    neck_dim = 15
    neck_activation = 'tanh'
    save_name = f'facescrub64_{model_name}_neck{neck_dim}{neck_activation}.pth'
    dataset_path = '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
    experiment_dir = f'../result_classifier/train_facescrub64_{model_name}_neck{neck_dim}{neck_activation}'
    backbone_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

    batch_size = 128
    epoch_num = 100

    device_ids_str = '2'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = IR152_64(num_classes, backbone_path=backbone_path)
    model = NeckWrapper(model, neck_dim=neck_dim, neck_activation=neck_activation)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # prepare dataset

    train_dataset = FaceScrub(
        dataset_path,
        train=True,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = FaceScrub(
        dataset_path,
        train=False,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose([ToTensor()]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=8,
    )

    # prepare train config

    loss_fn = 'ce'
    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn=loss_fn,
        save_per_epochs=1,
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
