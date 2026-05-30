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

from modelinversion.models import TorchvisionClassifierModel, VibWrapper, BiDOWrapper, LoraWrapper, auto_classifier_from_pretrained
from modelinversion.train import SimpleTrainConfig, SimpleTrainer
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss, freeze_front_layers, InverseFocalLoss
from modelinversion.datasets import FaceScrub

if __name__ == '__main__':

    num_classes = 530
    model_name = 'resnet152'


    src_path = '/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224_nopretrain/result_classifier/train_facescrub224_resnet152_growlora1_20/facescrub224_resnet152_growlora1_20.pth'
    focal_p = 4
    lr=0.002
    ft_epoch = 5
    alpha=1
    ls_val = -0.01
    freeze_ratio = 0.5

    root, folder, name = src_path.rsplit('/', 2)

    add_tag = f'_focal{focal_p}_alpha{alpha}_lr{lr}_ls{ls_val}_freeze{freeze_ratio}_{ft_epoch}'

    save_name = f'{name[:-4]}{add_tag}.pth'
    dataset_path = '/data/<usrname>/datasets/facescrub/'
    experiment_dir = os.path.join(root, f'{folder}{add_tag}')

    batch_size = 96
    epoch_num = ft_epoch

    device_ids_str = '6'
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

    # model = TorchvisionClassifierModel(
    #     model_name, num_classes=num_classes, weights='DEFAULT', register_last_feature_hook=True
    # )
    model = auto_classifier_from_pretrained(src_path).unwrap()
    freeze_front_layers(model, freeze_ratio)
    # state_dict = torch.load(backbone_path, map_location='cpu')["state_dict"]
    # del state_dict['model.fc.weight']
    # del state_dict['model.fc.bias']
    # load_res = model.load_state_dict(state_dict, strict=False)
    # print(load_res)
    # model = LoraWrapper(model, lora_dim=lora_dim)
    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(ft_epoch*0.6), int(ft_epoch*0.8
            
        )
        ], gamma=0.1
    )

    # prepare dataset

    train_dataset = FaceScrub(
        dataset_path,
        train=True,
        crop_center=False,
        preprocess_resolution=224,
        transform=Compose(
            [
                ToTensor(),
                # RandomResizedCrop(
                #     size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
                # ),
                # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    test_dataset = FaceScrub(
        dataset_path,
        train=False,
        crop_center=False,
        preprocess_resolution=224,
        transform=Compose(
            [
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

    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn=InverseFocalLoss(focal_p, alpha=alpha, label_smoothing=ls_val),
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader, save_best_ckpts=False)
