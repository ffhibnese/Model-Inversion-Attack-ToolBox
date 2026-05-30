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
    CenterCrop,
    Resize,
)

from modelinversion.models import (
    TorchvisionClassifierModel,
    LoktGenerator64,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
    get_stylegan2ada_generator,
    MultiHeadWrapper,
)
from modelinversion.train import SmileTrainConfig, SmileTrainer
from modelinversion.utils import Logger
from modelinversion.datasets import FaceScrub64, GeneratorDataset


def main(tag, cuda):

    model_pretrain_path = "/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/facescrub64/facescrub64_ir152.pth"
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    # if tag == 'no':
    #     tag = ''
    # else:
    tag = '_' + tag

    num_classes = 530
    # model_name = 'densenet121'
    save_name = f'facescrub64_ir152.pth'
    train_dataset_path = f'./dataset/plg_ffhq64_facescrub64_ir152_{tag}_dataset'
    test_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    experiment_dir = f'./classifier2/lokt_ffhq64_facescrub64_ir152{tag}/ir152'
    stylegan2ada_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/stylegan2-ffhq-256x256.pkl'
    # backbone_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

    batch_size = 128
    epoch_num = 100

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
    _, generator = get_stylegan2ada_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, single_w=True, optimize_in_w=False
    )
    generator = generator.to(device)
    generator.eval()

    # prepare target model

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)

    model = auto_classifier_from_pretrained(model_pretrain_path)
    model = MultiHeadWrapper(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[75, 90], gamma=0.1
    # )
    # lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epoch_num
    # )
    lr_schedular = None

    # prepare dataset

    transform = Compose(
        [
            CenterCrop((176, 176)),
            Resize((64, 64)),
            # RandomApply(
            #     [
            #         RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            #         # RandomApply([ColorJitter(brightness=0.2, contrast=0.2)]),
            #         RandomHorizontalFlip(),
            #         # RandomRotation(5),
            #     ]
            # ),
        ]
    )

    train_dataset: GeneratorDataset = GeneratorDataset.from_precreate(
        save_path=train_dataset_path,
        generator=generator,
        device=device,
        transform=transform,
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

    config = SmileTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        teacher=target_model,
    )

    trainer = SmileTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)

    logger.close()


for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    main(tag, 4)
