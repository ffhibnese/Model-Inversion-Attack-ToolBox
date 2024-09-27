import sys
import os
import time

sys.path.append("../../../src")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    SimpleGenerator64,
    GmiDiscriminator64,
    auto_classifier_from_pretrained,
)
from modelinversion.models.adapters.c2f import C2fThreeLayerMlpOutputMapping
from modelinversion.train import GmiGanTrainer, GmiGanTrainConfig, train_mapping_model
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, CelebA64

if __name__ == '__main__':

    target_model_ckpt_path = '/mnt/data/yhy/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/facescrub64/facescrub64_ir152_98.25.pth'
    embed_model_ckpt_path = '/mnt/data/yhy/Model-Inversion-Attack-ToolBox/checkpoints_v2/attacks/c2f/embedder/casia_incv1.pth'
    dataset_path = '/mnt/data/yhy/Model-Inversion-Attack-ToolBox/dataset/ffhq64'

    dataset_map_name = 'ffhq64_facescrub64'
    target_name = 'ir152'
    experiment_dir = f'../../../results_mapping/c2f/{dataset_map_name}/{target_name}'

    batch_size = 256

    device_ids_str = '3'

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model.eval()

    embed_model = auto_classifier_from_pretrained(embed_model_ckpt_path)
    embed_model = nn.DataParallel(embed_model, device_ids=gpu_devices).to(device)
    embed_model.eval()
    # print(target_model.training)
    # exit()

    # prepare dataset

    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        dataset_path,
        transform=ToTensor(),
    )
    # dataset = CelebA64(dataset_path, ToTensor())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # sampler=InfiniteSamplerWrapper(dataset),
    )

    mapping = C2fThreeLayerMlpOutputMapping(
        target_model.module.num_classes, 4096, embed_model.module.num_classes
    )
    mapping = nn.DataParallel(mapping).to(device)
    mapping.train()

    optimizer = torch.optim.Adam(mapping.parameters(), lr=0.001)
    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    train_mapping_model(
        40,
        mapping,
        optimizer,
        target_model,
        embed_model,
        dataloader,
        device=device,
        save_path=os.path.join(experiment_dir, 'mapping.pth'),
        schedular=optim_scheduler,
    )
