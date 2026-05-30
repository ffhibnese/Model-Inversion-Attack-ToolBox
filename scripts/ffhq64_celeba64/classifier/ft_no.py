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

from modelinversion.models import TorchvisionClassifierModel
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss
from modelinversion.datasets import CelebA64

if __name__ == '__main__':

    num_classes = 1000
    model_name = 'resnet18'
    save_name = f'celeba64_{model_name}.pth'
    dataset_path = '/mnt/data/<usrname>/datasets/pre_celeba_low'
    experiment_dir = f'../result_classifier/train_celeba64_{model_name}'
    backbone_path = '/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq64_facescrub64/result_classifier/pretrain_resnet18/pretrain_resnet18.pth'

    batch_size = 128
    epoch_num = 100

    device_ids_str = '4'
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

    # model = VGG16_64(num_classes)
    model = TorchvisionClassifierModel(
        model_name, num_classes=num_classes, resolution=64
    )
    state_dict = torch.load(backbone_path, map_location='cpu')['state_dict']
    del state_dict['model.fc.weight']
    del state_dict['model.fc.bias']
    load_info = model.load_state_dict(state_dict, strict=False)
    print(load_info)
    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )
    # lr_schedular = None

    # prepare dataset

    train_dataset = CelebA64(
        os.path.join(dataset_path, 'private_train'),
        # train=True,
        output_transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = CelebA64(
        os.path.join(dataset_path, 'private_test'),
        # train=True,
        output_transform=Compose(
            [
                ToTensor(),
                # RandomHorizontalFlip(p=0.5),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=16,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=16,
    )

    # prepare train config

    # loss_fn = LabelSmoothingCrossEntropyLoss(ls_val)

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

    trainer.train(epoch_num, train_loader, test_loader)
