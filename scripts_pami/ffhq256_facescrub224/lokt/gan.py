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

from modelinversion.models import (
    IR152_64,
    LoktGenerator256,
    LoktDiscriminator256,
    auto_classifier_from_pretrained,
)
from modelinversion.train import LoktGanTrainer, LoktGanTrainConfig
from modelinversion.utils import Logger, set_random_seed
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64


import torchvision.models


def main(tag, cuda):

    if tag == 'no':
        tag = ''
    else:
        tag = '_' + tag

    num_classes = 530
    target_model_ckpt_path = f'/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    dataset_path = '/mnt/data/<usrname>/datasets/ffhq256'
    experiment_dir = f'./gan/lokt_ffhq256_facescrub224_ir152_{tag}_gan'

    batch_size = 64
    max_iters = 105000

    device_ids_str = f'{cuda}'

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    set_random_seed(46)

    # prepare target models

    # target_model = IR152_64(num_classes=num_classes)
    # target_model.load_state_dict(
    #     torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    # )
    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    target_model = target_model.to(device)
    target_model.eval()

    # prepare dataset

    dataset = ImageFolder(
        dataset_path,
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

    generator = LoktGenerator256(num_classes, dim_z=z_dim)
    discriminator = LoktDiscriminator256(num_classes)

    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )

    train_config = LoktGanTrainConfig(
        experiment_dir=experiment_dir,
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


# main('rolss0.0_2', 0)

for tag in [
    'no',
    # 'vib0.01',
    # 'bido0.01_0.1_pretrain',
    # 'ls0.05',
    # 'tl0.5',
    # 'rolss0.0_2',
]:
    main(tag, 6)
