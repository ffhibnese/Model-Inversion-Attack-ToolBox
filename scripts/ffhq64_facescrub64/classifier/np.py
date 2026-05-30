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

from modelinversion.models import IR152_64, LoraWrapper, auto_classifier_from_pretrained
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, InverseFocalLoss, freeze_front_layers
from modelinversion.datasets import FaceScrub


def main(focal_p, cuda_idx):

    num_classes = 530
    model_name = 'ir152'
    # focal_p = 8
    # save_name = f'facescrub64_{model_name}_lora{lora_dim}_focal{focal_p}.pth'
    dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    # experiment_dir = f'../result_classifier/train_facescrub64_{model_name}_lora{lora_dim}_focal{focal_p}'

    lr = 0.001
    ft_epoch = 5

    origin_ckpt_path = f'../result_classifier/train_facescrub64_ir152_neck30tanh/facescrub64_ir152_neck30tanh.pth'

    root, folder, name = origin_ckpt_path.rsplit('/', 2)

    add_tag = f'_focal{focal_p}_lr{lr}_{ft_epoch}'

    save_name = f'{name[:-4]}{add_tag}.pth'
    experiment_dir = os.path.join(root, f'{folder}{add_tag}')

    batch_size = 128
    epoch_num = ft_epoch

    device_ids_str = f'{cuda_idx}'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = auto_classifier_from_pretrained(origin_ckpt_path)
    freeze_front_layers(model, 0.95)
    model = model.to(device)

    # for p in model.parameters():
    #     p.requires_grad = False

    # model.freeze_to_train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 4], gamma=0.1
    )

    # prepare dataset

    train_dataset = FaceScrub(
        dataset_path,
        train=True,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = FaceScrub(
        dataset_path,
        train=False,
        crop_center=True,
        preprocess_resolution=64,
        transform=Compose([ToTensor()]),
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

    loss_fn = InverseFocalLoss(gamma=focal_p)
    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn=loss_fn,
    )

    trainer = SimpleTrainer(config)

    print(trainer._test_loop(test_loader))

    trainer.train(epoch_num, train_loader, test_loader, save_best_ckpts=False)

    logger.close()


if __name__ == '__main__':
    ps = [16, 32]
    for p in ps:
        main(p, 5)
