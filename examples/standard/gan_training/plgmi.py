import sys
import os
import time

sys.path.append('../../../src')

import kornia
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import IR152_64, PlgmiGenerator64, PlgmiDiscriminator64
from modelinversion.train import PlgmiGanTrainer
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA

if __name__ == '__main__':

    top_k = 30
    num_classes = 1000
    target_model_ckpt_path = '<fill it>'
    dataset_path = '<fill it>'
    experiment_dir = '<fill it>'

    batch_size = 64
    max_iters = 150000

    device_ids_str = '0'

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = IR152_64(num_classes=num_classes)
    target_model.load_state_dict(
        torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    )
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)

    # prepare dataset

    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img

    dataset = CelebA(
        dataset_path,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose([ToTensor(), _noise_adder]),
    )
    dataloader = iter(
        DataLoader(
            dataset, batch_size=batch_size, sampler=InfiniteSamplerWrapper(dataset)
        )
    )

    # prepare GANs

    z_dim = 128

    generator = PlgmiGenerator64(num_classes, dim_z=z_dim)
    discriminator = PlgmiDiscriminator64(num_classes)

    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )

    # prepare trainer

    data_augment = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop(
            (64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)
        ),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    trainer = PlgmiGanTrainer(
        experiment_dir=experiment_dir,
        batch_size=batch_size,
        input_size=z_dim,
        generator=generator,
        discriminator=discriminator,
        num_classes=num_classes,
        target_model=target_model,
        classification_loss_fn='max_margin',
        device=device,
        augment=data_augment,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        save_ckpt_iters=1000,
        show_images_iters=1000,
        show_train_info_iters=100,
    )

    # train gan

    trainer.train(dataloader, max_iters)
