import sys
import os

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import (
    LoktGenerator64,
    IR152_64,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
    get_stylegan2ada_generator,
)
from modelinversion.datasets import (
    generator_generate_datasets,
    preprocess_facescrub_fn,
    GeneratorDataset,
)


def main(tag):

    paths = get_attack_paths('p2i', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 100

    batch_size = 100

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # dataset generator

    z_dim = 512

    _, generator = get_stylegan2ada_generator(
        paths.stylegan2ada_path, paths.stylegan2ada_ckpt_path, single_w=True, optimize_in_w=False
    )
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path).to(device)
    target_model.eval()

    gan_to_target_transform = TF.Compose(
        [TF.CenterCrop((176, 176)), TF.Resize((64, 64))]
    )

    dataset = GeneratorDataset.create(
        z_dim,
        num_classes=num_classes,
        generate_num_per_class=30,
        generator=generator,
        target_model=target_model,
        batch_size=batch_size,
        device=device,
        gan_to_target_transform=gan_to_target_transform,
        num_per_class_for_selection=200,
        save_confidence=True,
    )

    dataset.save(paths.p2i_fake_dataset_path)


for tag in ALL_TAGS:
    main(tag)
