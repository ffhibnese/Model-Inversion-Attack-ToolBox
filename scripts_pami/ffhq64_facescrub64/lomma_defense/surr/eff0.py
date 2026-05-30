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

from modelinversion.models import (
    auto_classifier_from_pretrained,
    EfficientNet_b0_64,
    EfficientNet_b1_64,
    EfficientNet_b2_64,
)
from modelinversion.train import DistillTrainer, DistillTrainConfig
from modelinversion.utils import Logger, freeze
from modelinversion.datasets import FaceScrub64


def main(tag, model_name, device_ids_available):

    num_classes = 530
    # model_name = 'efficientnet_b2'
    teacher_name = 'ir152'
    save_name = f'ffhq64_{model_name}_facescrub64_{tag}.pth'
    train_dataset_path = '/mnt/data/<usrname>/datasets/ffhq64'
    test_dataset_path = (
        # '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
        '/mnt/data/<usrname>/datasets/facescrub/'
    )
    experiment_dir = f'./distill_ffhq64_{model_name}_facescrub64_{tag}'
    teacher_ckpt_path = f'../../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    batch_size = 128
    epoch_num = 100

    # device_ids_available = '1'
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

    model_cls = (
        EfficientNet_b0_64
        if model_name == 'efficientnet_b0'
        else (
            EfficientNet_b1_64
            if model_name == 'efficientnet_b1'
            else EfficientNet_b2_64
        )
    )

    model = model_cls(num_classes, pretrained=True)
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

    logger.close()


tags = ['rolss0.0_2']

modelnames = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']

for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    # main(tag, 6)
    for model_name in modelnames:
        main(tag, model_name, '5')

# for i, tag in enumerate(tags):
#     for j, model_name in enumerate(modelnames):
#         idx = i * len(modelnames) + j
#         if idx == 2:
#             main(tag, model_name, '1')
#             exit()
#         else:
#             if os.fork() == 0:
#                 main(tag, model_name, '1')
#                 exit()
