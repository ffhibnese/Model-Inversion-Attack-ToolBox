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

from modelinversion.models import IR152_64
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper

if __name__ == '__main__':

    num_classes = 1000
    model_name = 'ir152'
    save_name = f'{model_name}.pth'
    train_dataset_path = '<fill it>'
    test_dataset_path = '<fill it>'
    experiment_dir = f'<fill it>'
    backbone_path = '<fill it>'

    batch_size = 128
    epoch_num = 100

    device_ids_str = '0'
    pin_memory = True

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
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    lr_schedular = None

    # prepare dataset

    train_dataset = ImageFolder(
        train_dataset_path,
        Compose(
            [
                Resize((64, 64), antialias=True),
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = ImageFolder(
        test_dataset_path,
        Compose(
            [
                Resize((64, 64), antialias=True),
                ToTensor(),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    # prepare train config

    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn='cross_entropy',
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)