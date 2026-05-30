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
from torchvision.transforms import ToTensor, Compose

from modelinversion.models import (
    auto_classifier_from_pretrained,
    KedmiDiscriminator256,
    SimpleGenerator256,
)
from modelinversion.train import KedmiGanTrainer, KedmiGanTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64


def main(tag):
    paths = get_attack_paths('ked', tag)

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

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model.eval()
    # print(target_model.training)
    # exit()

    # prepare dataset

    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        paths.eval_dataset_path,
        transform=ToTensor(),
    )
    # dataset = CelebA64(dataset_path, ToTensor())
    dataloader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=8,
        )
    )

    # prepare GANs

    z_dim = 100

    generator = SimpleGenerator256(in_dim=z_dim)
    discriminator = KedmiDiscriminator256(num_classes)

    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    # prepare trainer

    config = KedmiGanTrainConfig(
        experiment_dir=paths.gan_experiment_dir,
        batch_size=batch_size,
        input_size=z_dim,
        generator=generator,
        discriminator=discriminator,
        target_model=target_model,
        device=device,
        augment=None,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        save_ckpt_iters=1000,
        show_images_iters=1000,
        show_train_info_iters=100,
    )

    trainer = KedmiGanTrainer(config)
    # train gan

    trainer.train(dataloader, max_iters)

    logger.close()


all_pids = []

for tag in ALL_TAGS:
    main(tag)
#     if pid := os.fork():
#         all_pids.append(pid)
#         main(tag)
#         exit()

# for pid in all_pids:
#     os.waitpid(pid, 0)
