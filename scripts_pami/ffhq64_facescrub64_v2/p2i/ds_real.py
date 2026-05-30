import sys
import os
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

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
    preprocess_celebs_fn,
    preprocess_facescrub_fn,
)


def main(tag):

    paths = get_attack_paths('p2i', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    top_k = 30
    num_classes = 530
    loradim = 1

    batch_size = 50

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path).to(device)
    target_model.eval()

    # dataset generation

    if not os.path.exists(paths.p2i_real_dataset_path):

        top_k_selection(
            top_k=top_k,
            src_dataset_path=paths.eval_dataset_path,
            dst_dataset_path=paths.p2i_real_dataset_path,
            batch_size=batch_size,
            target_model=target_model,
            num_classes=num_classes,
            device=device,
            create_aug_images_fn=lambda img: [img],
        )


for tag in ALL_TAGS:
    main(tag)
