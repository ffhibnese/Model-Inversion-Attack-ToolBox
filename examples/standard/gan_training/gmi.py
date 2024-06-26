import sys
import os
import time

sys.path.append("../../../src")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import SimpleGenerator64, GmiDiscriminator64
from modelinversion.train import GmiGanTrainer, GmiGanTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64

if __name__ == "__main__":

    dataset_path = '<fill it>'
    experiment_dir = '<fill it>'

    batch_size = 64
    max_iters = 150000

    device_ids_available = "2"

    # prepare logger

    now_time = time.strftime(r"%Y%m%d_%H%M", time.localtime(time.time()))
    logger = Logger(experiment_dir, f"train_gan_{now_time}.log")

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare dataset

    dataset = CelebA64(
        dataset_path,
        output_transform=ToTensor(),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=InfiniteSamplerWrapper(dataset)
    )

    # prepare GANs

    z_dim = 100

    generator = SimpleGenerator64(in_dim=z_dim)
    discriminator = GmiDiscriminator64()

    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    # prepare trainer

    train_config = GmiGanTrainConfig(
        experiment_dir=experiment_dir,
        # train args
        batch_size=batch_size,
        input_size=z_dim,
        generator=generator,
        discriminator=discriminator,
        device=device,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        # log args
        save_ckpt_iters=1000,
        show_images_iters=1000,
        show_train_info_iters=100,
    )

    trainer = GmiGanTrainer(train_config)

    # train gan

    trainer.train(dataloader, max_iters)
