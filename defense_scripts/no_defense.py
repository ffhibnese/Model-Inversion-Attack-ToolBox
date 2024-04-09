import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    RandomHorizontalFlip,
    Compose,
    ToPILImage,
    Lambda,
    Resize,
)
from torchvision import transforms as tvtrans

from development_config import get_dirs
from modelinversion.defense import *
from modelinversion.models import get_model
from modelinversion.foldermanager import FolderManager
from modelinversion.utils import RandomIdentitySampler
import math

if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    defense_type = 'no_defense'

    model_name = 'swin_b'

    dirs = get_dirs(f'{defense_type}_{model_name}')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
        dirs['defense_ckpt_dir'],
    )

    folder_manager = FolderManager(
        ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type
    )

    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.03
    device = 'cuda'
    batch_size = 64

    args = BaseTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='no_defense',
        tqdm_strategy=TqdmStrategy.ITER,
        device=device,
    )

    model = get_model(model_name, dataset_name, device, backbone_pretrain=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num)

    trainer = RegTrainer(args, folder_manager, model, optimizer, scheduler)

    trainset = ImageFolder(
        './dataset/celeba/split/private/train',
        transform=Compose([RandomHorizontalFlip(p=0.5), ToTensor()]),
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )

    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(
        testset, batch_size, shuffle=False, pin_memory=True, drop_last=True
    )

    trainer.train(trainloader, testloader)
