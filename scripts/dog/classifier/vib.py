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
    TorchvisionClassifierModel,
    VibWrapper,
    BiDOWrapper,
    LoraWrapper,
)
from modelinversion.train import VibTrainConfig, VibTrainer
from modelinversion.utils import (
    Logger,
    LabelSmoothingCrossEntropyLoss,
    freeze_front_layers,
)
from modelinversion.datasets import StanfordDogs

if __name__ == '__main__':
    resolution = 224

    num_classes = 530
    model_name = 'resnet152'
    save_name = f'stanford_dogs_{model_name}_vib.pth'
    dataset_path = '/mnt/data/<usrname>/datasets/stanford_dogs/'
    experiment_dir = f'../result_classifier/train_stanford_dogs_{model_name}_vib'
    backbone_path = '/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_stanford_dogs/result_classifier/pretrain_resnet152/pretrain_stanford_dogs_resnet152.pth'

    batch_size = 96
    epoch_num = 100

    device_ids_str = '7'
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

    model = TorchvisionClassifierModel(
        model_name,
        num_classes=num_classes,
        weights='DEFAULT',
        register_last_feature_hook=True,
    )
    # state_dict = torch.load(backbone_path, map_location='cpu')["state_dict"]
    # del state_dict['model.fc.weight']
    # del state_dict['model.fc.bias']
    # load_res = model.load_state_dict(state_dict, strict=False)
    # print(load_res)
    model = VibWrapper(model)
    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # prepare dataset

    train_dataset = StanfordDogs(
        dataset_path,
        train=True,
        transform=Compose(
            [
                Resize((resolution, resolution), antialias=True),
                ToTensor(),
                RandomResizedCrop(
                    size=(resolution, resolution),
                    scale=(0.85, 1),
                    ratio=(1, 1),
                    antialias=True,
                ),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    test_dataset = StanfordDogs(
        dataset_path,
        train=False,
        transform=Compose(
            [
                Resize((resolution, resolution), antialias=True),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
    )

    # prepare train config

    config = VibTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn='ce',
    )

    trainer = VibTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
