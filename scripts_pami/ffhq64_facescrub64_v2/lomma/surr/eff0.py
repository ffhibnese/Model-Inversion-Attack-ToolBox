import sys
import os
import time

sys.path.append('../../../../src')
sys.path.append('../../../..')
from attack_paths import get_attack_paths, ALL_TAGS

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

from modelinversion.models import (
    auto_classifier_from_pretrained,
    TorchvisionClassifierModel,
)
from modelinversion.train import DistillTrainer, DistillTrainConfig
from modelinversion.utils import Logger, freeze
from modelinversion.datasets import FaceScrub64


def main(tag, model_name, device_ids_available):
    # Use lomma_lgmi to get common config (distill models are same for both variants)
    paths = get_attack_paths('lomma_lgmi', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 530
    teacher_name = 'ir152'
    save_name = f'ffhq64_{model_name}_facescrub64_{tag}.pth'
    experiment_dir = f'./results_attack/distill_{model_name}_ir152_{tag}'

    batch_size = paths.distill_batch_size  # halved from 128
    epoch_num = paths.distill_epoch_num

    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    teacher = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    if hasattr(teacher, 'unwrap'):
        teacher = teacher.unwrap()
    teacher = teacher.to(device)
    teacher.eval()

    model = TorchvisionClassifierModel(
        model_name, num_classes, resolution=64, weights='DEFAULT'
    )
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    lr_schedular = None

    # prepare dataset

    train_dataset = ImageFolder(
        paths.lomma_public_dataset_path,
        transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    test_dataset = FaceScrub64(
        paths.eval_dataset_path,
        train=False,
        output_transform=Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
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

    logger.close()


for tag in ALL_TAGS:
    for model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
        main(tag, model_name, '4')
