import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from modelinversion.attack.Lokt.surrogate_trainer import (
    LoktSurrogateTrainArgs,
    LoktSurrogateTrainer,
)
from development_config import get_dirs
from modelinversion.models import *

from modelinversion.foldermanager import FolderManager

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':

    model_name = 'densenet121'

    dirs = get_dirs(f'lokt_surrogate_{model_name}')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
        dirs['defense_ckpt_dir'],
    )

    folder_manager = FolderManager(
        ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, 'no_defense'
    )

    dataset_name = 'celeba'
    epoch_num = 200
    lr = 0.01
    device = 'cuda'
    batch_size = 256

    train_args = LoktSurrogateTrainArgs(
        model_name,
        dataset_name,
        epoch_num,
        device=device,
        target_name='vgg16',
        target_dataset_name='celeba',
    )

    model = get_model(model_name, dataset_name, device=device, backbone_pretrain=True)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    trainer = LoktSurrogateTrainer(
        train_args,
        folder_manager=folder_manager,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    trainset = trainer.get_trainset()

    trainloader = DataLoader(trainset, batch_size, shuffle=True)

    trainer.train(trainloader, None)
