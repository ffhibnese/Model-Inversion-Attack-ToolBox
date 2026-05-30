import sys
import os
import time

sys.path.append('../../../src')

import kornia
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    IR152_64,
    PlgmiGenerator64,
    PlgmiDiscriminator64,
    auto_classifier_from_pretrained,
)
from modelinversion.train import PlgmiGanTrainer, PlgmiGanTrainConfig
from modelinversion.utils import Logger
from modelinversion.datasets import InfiniteSamplerWrapper, LabelImageFolder


import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import auto_classifier_from_pretrained
from modelinversion.datasets import (
    top_k_selection,
    preprocess_celeba_fn,
    preprocess_facescrub_fn,
)


def main(tag):

    top_k = 30
    num_classes = 530
    loradim = 1
    # tag = 'rolss0.0_2'
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'
    src_dataset_path = '/mnt/data/<usrname>/datasets/ffhq64'
    dst_dataset_path = f'./real_dataset/ffhq64_facescrub64_ir152_{tag}_dataset'
    experiment_dir = f'./results_gan/plg_ffhq64_facescrub64_ir152_{tag}_gan'

    dataset_path = dst_dataset_path

    batch_size = 50
    device_ids_str = '7'

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
    # target_model = nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.eval()

    # dataset generation

    # transform = TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if not os.path.exists(dst_dataset_path):

        top_k_selection(
            top_k=top_k,
            src_dataset_path=src_dataset_path,
            dst_dataset_path=dst_dataset_path,
            batch_size=batch_size,
            target_model=target_model,
            num_classes=num_classes,
            device=device,
            create_aug_images_fn=lambda img: [img],
        )


for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    main(tag)
