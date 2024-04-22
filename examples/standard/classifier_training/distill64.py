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

from modelinversion.models import IR152_64, EfficientNet_b0_64
from modelinversion.train import DistillTrainer, DistillTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import CelebA

if __name__ == '__main__':

    num_classes = 1000
    model_name = 'efficientnet_b0'
    teacher_name = 'ir152'
    save_name = f'celeba64_{model_name}.pth'
    train_dataset_path = '../../../test/celeba/public'
    test_dataset_path = '../../../test/celeba/private_test'
    experiment_dir = f'../../../results/distill_celeba64_{model_name}_{teacher_name}_v3'
    teacher_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    )

    batch_size = 128
    epoch_num = 100

    device_ids_available = '1'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    teacher = IR152_64(num_classes)
    teacher.load_state_dict(
        torch.load(teacher_ckpt_path, map_location='cpu')['state_dict']
    )
    teacher = teacher.to(device)

    model = EfficientNet_b0_64(num_classes, prtrained=True)
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    # prepare train config

    config = DistillTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        teacher=teacher,
    )

    trainer = DistillTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
