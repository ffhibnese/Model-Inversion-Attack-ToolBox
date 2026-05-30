import sys
import os
import time

sys.path.append("../../../src")
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    auto_classifier_from_pretrained,
)
from modelinversion.models.adapters.c2f import C2fThreeLayerMlpOutputMapping
from modelinversion.train import GmiGanTrainer, GmiGanTrainConfig, train_mapping_model
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64


def main(tag):
    paths = get_attack_paths('c2f', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    batch_size = 128  # halved from 256

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model.eval()

    embed_model = auto_classifier_from_pretrained(paths.cls_embed_model_ckpt_path)
    embed_model = nn.DataParallel(embed_model, device_ids=gpu_devices).to(device)
    embed_model.eval()

    # prepare dataset

    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        paths.eval_dataset_path,
        transform=ToTensor(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    mapping = C2fThreeLayerMlpOutputMapping(
        target_model.module.num_classes, 4096, embed_model.module.num_classes
    )
    mapping = nn.DataParallel(mapping).to(device)
    mapping.train()

    optimizer = torch.optim.Adam(mapping.parameters(), lr=0.001)
    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    train_mapping_model(
        40,
        mapping,
        optimizer,
        target_model,
        embed_model,
        dataloader,
        device=device,
        save_path=os.path.join(paths.experiment_dir, 'mapping.pth'),
        schedular=optim_scheduler,
    )

    logger.close()


for tag in ALL_TAGS:
    main(tag)
