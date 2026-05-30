import sys
import os
import time

sys.path.append('../../../src')
sys.path.append('..')
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
    RandomRotation,
    RandomApply,
)

from modelinversion.models import (
    TorchvisionClassifierModel,
    LoktGenerator64,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import FaceScrub224, GeneratorDataset


def main(tag, cuda, model_name):
    paths = get_attack_paths('lokt', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 530
    save_name = f'facescrub224_{model_name}.pth'
    train_dataset_path = paths.lokt_train_dataset_path
    test_dataset_path = paths.eval_dataset_path
    experiment_dir = f'./classifier/lokt_ffhq256_facescrub224_ir152{{tag_str}}/{model_name}'.format(
        tag_str='' if tag == 'no' else f'_{tag}'
    )

    batch_size = paths.cls_train_batch_size  # halved from 128
    epoch_num = 10

    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare generator
    z_dim = 128
    generator = auto_generator_from_pretrained(paths.generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target model

    model = TorchvisionClassifierModel(
        model_name, num_classes, resolution=224, weights='DEFAULT'
    )
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch_num
    )

    # prepare dataset

    train_dataset: GeneratorDataset = GeneratorDataset.from_precreate(
        save_path=train_dataset_path,
        generator=generator,
        device=device,
        transform=RandomApply(
            [
                RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                RandomApply([ColorJitter(brightness=0.2, contrast=0.2)]),
                RandomHorizontalFlip(),
                RandomRotation(5),
            ]
        ),
    )
    test_dataset = FaceScrub224(
        test_dataset_path,
        train=False,
        output_transform=Compose([ToTensor()]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # prepare train config

    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn='cross_entropy',
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)

    logger.close()


for tag in ALL_TAGS:
    all_pids = []
    for model_name in ['densenet121', 'densenet169', 'densenet161']:
        main(tag, 3, model_name)