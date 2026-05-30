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
from modelinversion.datasets import FaceScrub64, GeneratorDataset


def main(tag, cuda, model_name):

    # if tag == 'no':
    #     tag = ''
    # else:
    tag = '_' + tag

    num_classes = 530
    # model_name = 'densenet121'
    save_name = f'facescrub64_{model_name}.pth'
    train_dataset_path = (
        f'./dataset/lokt_ffhq64_facescrub64_ir152{tag}_dataset/dataset.pt'
    )
    test_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    experiment_dir = f'./classifier/lokt_ffhq64_facescrub64_ir152{tag}/{model_name}'
    generator_ckpt_path = f'./gan/lokt_ffhq64_facescrub64_ir152{tag}_gan/G.pth'
    # backbone_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

    batch_size = 128
    epoch_num = 10

    device_ids_str = str(cuda)
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare generator
    z_dim = 128
    # generator = LoktGenerator64(num_classes, dim_z=z_dim)
    # generator.load_state_dict(
    #     torch.load(generator_ckpt_path, map_location='cpu')['state_dict']
    # )
    generator = auto_generator_from_pretrained(generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target model

    model = TorchvisionClassifierModel(
        model_name, num_classes, resolution=64, weights='DEFAULT'
    )
    model = nn.DataParallel(model, device_ids=gpu_devices).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[75, 90], gamma=0.1
    # )
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
                RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                RandomApply([ColorJitter(brightness=0.2, contrast=0.2)]),
                RandomHorizontalFlip(),
                RandomRotation(5),
            ]
        ),
    )
    # train_dataset = CelebA(
    #     train_dataset_path,
    #     crop_center=False,
    #     preprocess_resolution=64,
    #     transform=Compose(
    #         [
    #             ToTensor(),
    #             RandomApply(
    #                 [
    #                     RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    #                     RandomApply([ColorJitter(brightness=0.2, contrast=0.2)]),
    #                     RandomHorizontalFlip(),
    #                     RandomRotation(5),
    #                 ]
    #             ),
    #         ]
    #     ),
    # )
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


# tags = [
#     'no',
#     'neck30tanh_focal8_lr1_5',
#     'tl0.5',
#     'bido0.01_0.1',
#     'ls_-0.3',
#     # -------
#     'vib0.1',
#     'rolss0.0_2',
# ]


# tag = tags[6]
# cuda = 2

# if os.fork() == 0:
#     main(tag, cuda, 'densenet121')

#     exit()

# if os.fork() == 0:
#     main(tag, cuda, 'densenet169')

#     exit()

# main(tag, cuda, 'densenet161')

for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    all_pids = []
    for model_name in ['densenet121', 'densenet169', 'densenet161']:
        cuda = 2
        if pid := os.fork() == 0:
            main(tag, cuda, model_name)
            exit()

        all_pids.append(pid)

    for pid in all_pids:
        os.waitpid(pid, 0)
