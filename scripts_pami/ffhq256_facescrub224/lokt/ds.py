import sys
import os

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import (
    LoktGenerator256,
    IR152_64,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.datasets import (
    generator_generate_datasets,
    preprocess_facescrub_fn,
    GeneratorDataset,
)


def main(tag):
    paths = get_attack_paths('lokt', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 530
    batch_size = paths.ds_batch_size  # halved from 200

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    # dataset generator

    z_dim = 128

    generator = auto_generator_from_pretrained(paths.generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path).to(device)
    target_model.eval()

    dataset = GeneratorDataset.create(
        z_dim,
        num_classes=num_classes,
        generate_num_per_class=500,
        generator=generator,
        target_model=target_model,
        batch_size=batch_size,
        device=device,
    )

    dataset.save(paths.lokt_train_dataset_path)


for tag in ALL_TAGS:
    main(tag)