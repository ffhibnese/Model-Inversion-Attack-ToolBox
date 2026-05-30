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

from modelinversion.models import (
    IR152_64,
    LoraWrapper,
    auto_classifier_from_pretrained,
    LRCWrapper,
)
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, InverseFocalLoss, freeze_front_layers
from modelinversion.datasets import FaceScrub


def main(ratio, cuda_device, print_terminal=False):

    num_classes = 530
    model_name = 'ir152'
    focal_p = 8
    # save_name = f'facescrub64_{model_name}_lora{lora_dim}_focal{focal_p}.pth'
    dataset_path = '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
    # experiment_dir = f'../result_classifier/train_facescrub64_{model_name}_lora{lora_dim}_focal{focal_p}'

    lr = 0.01
    ft_epoch = 100

    origin_ckpt_path = f'/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/facescrub64_sim/facescrub64_ir152.pth'

    root, folder, name = origin_ckpt_path.rsplit('/', 2)

    add_tag = f'_lrc{ratio}'

    tag = f'{name[:-4]}{add_tag}'
    save_name = f'{tag}.pth'
    # experiment_dir = os.path.join(root, f'{folder}{add_tag}')
    root = "../result_classifier"
    experiment_dir = os.path.join(root, f'train_{tag}')

    batch_size = 128
    epoch_num = ft_epoch

    device_ids_str = str(cuda_device)
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(
        experiment_dir, f'train_gan_{now_time}.log', print_terminal=print_terminal
    )

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = auto_classifier_from_pretrained(origin_ckpt_path)
    # freeze_front_layers(model, 0.95)
    model = LRCWrapper(model, keep_rank_or_ratio=ratio)
    # nn.Linear
    # model.compression()
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

    # loss_fn = InverseFocalLoss(gamma=focal_p)
    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn='ce',
    )

    trainer = SimpleTrainer(config)

    print(trainer._test_loop(test_loader))

    trainer.train(epoch_num, train_loader, test_loader, save_best_ckpts=True)

    logger.close()


if __name__ == '__main__':
    # main(0.8)
    ratios = [
        0.3,
        0.34,
        0.38,
        0.42,
        0.46,
        0.5,
        0.54,
        0.58,
        0.62,
        0.66,
        0.7,
        0.74,
        0.78,
        0.82,
        0.86,
        0.9,
        0.94,
        0.98,
        # 0.82,
        # 0.84,
        # 0.86,
        # 0.88,
        # 0.9,
        # 0.92,
        # 0.94,
        # 0.96,
        # 0.98,
        # #
        # 0.62,
        # 0.64,
        # 0.66,
        # 0.68,
        # 0.7,
        # 0.72,
        # 0.74,
        # 0.76,
        # 0.78,
    ]

    if os.fork() == 0:
        for i in range(3):
            main(ratios[i], 0)
        exit(0)

    if os.fork() == 0:
        for i in range(3, 6):
            main(ratios[i], 0)
        exit(0)

    if os.fork() == 0:
        for i in range(6, 9):
            main(ratios[i], 0)
        exit(0)

    if os.fork() == 0:
        for i in range(9, 12):
            main(ratios[i], 1)
        exit(0)

    if os.fork() == 0:
        for i in range(12, 15):
            main(ratios[i], 1)
        exit(0)

    for i in range(15, 18):
        main(ratios[i], 1, True)
