import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import torch
from torch import nn
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

from development_config import get_dirs
from modelinversion.defense import *
from modelinversion.models import get_model
from modelinversion.foldermanager import FolderManager
from modelinversion.utils import RandomIdentitySampler

if __name__ == '__main__':

    defense_type = 'dp'

    dirs = get_dirs(defense_type)
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

    model_name = 'vgg16'
    dataset_name = 'celeba'
    epoch_num = 200
    lr = 0.03
    device = 'cuda:0'
    batch_size = 64

    args = DPTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type=defense_type,
        tqdm_strategy=TqdmStrategy.ITER,
        device=device,
        noise_multiplier=0.01,
        clip_grad_norm=1,
    )

    model = get_model(model_name, dataset_name, device, backbone_pretrain=True)
    # model.bn.bias.requires_grad_(True)
    # model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )

    trainer = DPTrainer(args, folder_manager, model, optimizer, None)

    trainset = ImageFolder(
        './dataset/celeba/split/private/train',
        transform=Compose([RandomHorizontalFlip(p=0.5), ToTensor()]),
    )

    train_sampler = RandomIdentitySampler(trainset, batch_size, 4)
    trainloader = DataLoader(
        trainset, batch_size, sampler=train_sampler, pin_memory=True, drop_last=True
    )

    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(
        testset, batch_size, shuffle=False, pin_memory=True, drop_last=True
    )

    trainer.train(trainloader, testloader)
