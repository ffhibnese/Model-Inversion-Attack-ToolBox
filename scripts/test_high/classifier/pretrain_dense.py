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

from modelinversion.models import TorchvisionClassifierModel
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss
from modelinversion.datasets import LabelImageFolder

if __name__ == '__main__':

    num_classes = 86000
    model_name = 'densenet169'
    save_name = f'pretrain_{model_name}.pth'
    dataset_path = '/data/<usrname>/datasets/msceleb1m/imgs'
    experiment_dir = f'../result_classifier/pretrain_{model_name}'

    batch_size = 128
    epoch_num = 5

    device_ids_str = '3'
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

    model = TorchvisionClassifierModel(
        model_name, num_classes=num_classes, weights='DEFAULT'
    )
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    # exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 4], gamma=0.1
    )

    # prepare dataset

    train_dataset = LabelImageFolder(
        dataset_path,
        transform=Compose(
            [
                Resize((224, 224), antialias=True),
                ToTensor(),
                RandomResizedCrop(
                    size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
                ),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=16,
    )

    # prepare train config

    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        save_per_epochs=1,
        loss_fn='ce',
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, None)
