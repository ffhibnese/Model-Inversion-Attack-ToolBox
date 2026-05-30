import sys
import os
import time

sys.path.append('../../../../src')

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

from modelinversion.models import auto_classifier_from_pretrained, EfficientNet_b0_64
from modelinversion.train import DistillTrainer, DistillTrainConfig
from modelinversion.utils import Logger, freeze
from modelinversion.datasets import FaceScrub64

if __name__ == '__main__':

    num_classes = 530
    model_name = 'efficientnet_b0'
    teacher_name = 'ir152'
    save_name = f'ffhq64_{model_name}_facescrub64_{teacher_name}_tl0.4.pth'
    train_dataset_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/dataset/ffhq64'
    test_dataset_path = (
        '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
    )
    experiment_dir = f'./distill_ffhq64_{model_name}_facescrub64_{teacher_name}_tl0.4'
    teacher_ckpt_path = f'../../result_classifier/train_facescrub64_ir152_tl0.4/facescrub64_ir152_tl0.4.pth'

    batch_size = 128
    epoch_num = 100

    device_ids_available = '2'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    teacher = auto_classifier_from_pretrained(teacher_ckpt_path)
    if hasattr(teacher, 'unwrap'):
        teacher = teacher.unwrap()
    teacher = teacher.to(device)
    # freeze(teacher)
    teacher.eval()

    model = EfficientNet_b0_64(num_classes, pretrained=True)
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    lr_schedular = None

    # prepare dataset

    train_dataset = ImageFolder(
        train_dataset_path,
        transform=Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    test_dataset = FaceScrub64(
        test_dataset_path,
        train=False,
        output_transform=Compose([ToTensor()]),
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

    config = DistillTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        teacher=teacher,
    )

    trainer = DistillTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
