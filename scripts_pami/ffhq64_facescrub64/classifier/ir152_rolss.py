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

from modelinversion.models import IR152_64, SkipConnectionWrapper
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import (
    Logger,
    LabelSmoothingCrossEntropyLoss,
    freeze_front_layers,
)
from modelinversion.datasets import FaceScrub


def main(erase_ratio, keep_ratio, cuda_device):

    num_classes = 530
    model_name = 'ir152'
    # tl_coef = 0.4
    # keep_ratio = 0.0
    # erase_ratio = 0.3
    save_name = f'facescrub64_{model_name}_rolss{keep_ratio}_{erase_ratio}.pth'
    dataset_path = '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
    experiment_dir = f'../result_classifier/train_facescrub64_{model_name}_rolss{keep_ratio}_{erase_ratio}'
    backbone_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

    batch_size = 128
    epoch_num = 100

    device_ids_str = str(cuda_device)
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

    model = IR152_64(num_classes, backbone_path=backbone_path)
    # # freeze_front_layers(model, ratio=tl_coef)
    # for n in list(model.children()):
    #     print(n.__class__)
    #     for m in list(n.children()):
    #         print('> ', m.__class__)
    #         for p in list(m.children()):
    #             print('>> ', p.__class__)
    # exit()
    model = SkipConnectionWrapper(
        model, residule_keep_ratio=keep_ratio, erase_ratio_or_num=erase_ratio
    )
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )
    torch.optim.Adam
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
        num_workers=4,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
        prefetch_factor=2,
    )

    # prepare train config

    loss_fn = 'ce'

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

    trainer.train(epoch_num, train_loader, test_loader)

    logger.close()


# ratios = [1, 2, 3, 4, 5, 6]

# for i, r in enumerate(ratios[:-1]):
#     if os.fork() == 0:
#         main(r, i // 3)

# main(ratios[-1], 1)

main(3, 0.2, '3')
