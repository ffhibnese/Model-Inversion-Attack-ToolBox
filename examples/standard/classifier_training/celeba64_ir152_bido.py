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

from modelinversion.models import IR152_64, BiDOWrapper
from modelinversion.train import BiDOTrainConfig, BiDOTrainer
from modelinversion.utils import Logger
from modelinversion.datasets import CelebA

if __name__ == '__main__':

    num_classes = 1000
    model_name = 'ir152'
    save_name = f'celeba64_{model_name}_bido_ih0.05_oh2.pth'
    train_dataset_path = '<fill it>'
    test_dataset_path = '<fill it>'
    experiment_dir = '<fill it>'
    backbone_path = '../../../checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

    batch_size = 128
    epoch_num = 150

    device_ids_str = '2'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_classifier_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = IR152_64(
        num_classes, backbone_path=backbone_path, register_last_feature_hook=True
    )
    model = BiDOWrapper(model)
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 100, 125, 140], gamma=0.3
    )
    lr_schedular = None

    # prepare dataset

    train_dataset = CelebA(
        train_dataset_path,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = CelebA(
        test_dataset_path,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose([ToTensor()]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
    )

    # prepare train config

    config = BiDOTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn='cross_entropy',
        coef_hidden_input=0.05,
        coef_hidden_output=2,
    )

    trainer = BiDOTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
