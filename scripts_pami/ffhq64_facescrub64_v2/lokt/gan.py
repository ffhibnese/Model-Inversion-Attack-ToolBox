import sys
import os
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import kornia
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose

from modelinversion.models import (
    IR152_64,
    LoktGenerator64,
    LoktDiscriminator64,
    auto_classifier_from_pretrained,
)
from modelinversion.train import LoktGanTrainer, LoktGanTrainConfig
from modelinversion.utils import Logger, set_random_seed
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64


import torchvision.models


def main(tag):
    paths = get_attack_paths('lokt', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 530
    batch_size = paths.gan_batch_size  # halved from 64
    max_iters = paths.gan_max_iters

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.gan_experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    set_random_seed(46)

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    target_model = target_model.to(device)
    target_model.eval()

    # prepare dataset

    dataset = ImageFolder(
        paths.eval_dataset_path,
        transform=Compose([ToTensor()]),
    )
    dataloader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=4,
        )
    )

    # prepare GANs

    z_dim = 128

    generator = LoktGenerator64(num_classes, dim_z=z_dim)
    discriminator = LoktDiscriminator64(num_classes)

    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )

    train_config = LoktGanTrainConfig(
        experiment_dir=paths.gan_experiment_dir,
        batch_size=batch_size,
        input_size=z_dim,
        generator=generator,
        discriminator=discriminator,
        num_classes=num_classes,
        target_model=target_model,
        classification_loss_fn='cross_entropy',
        device=device,
        augment=None,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        save_ckpt_iters=2000,
        start_class_loss_iters=5000,
        show_images_iters=2000,
        show_train_info_iters=473,
        class_loss_weight=1.5,
    )

    # train gan

    trainer = LoktGanTrainer(train_config)

    trainer.train(dataloader, max_iters)


for tag in ALL_TAGS:
    main(tag)
