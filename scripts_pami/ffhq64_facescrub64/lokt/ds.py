import sys
import os

sys.path.append('../../../src')

import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import (
    LoktGenerator64,
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

    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    tag = '_' + tag

    num_classes = 530
    generator_ckpt_path = f'./gan/lokt_ffhq64_facescrub64_ir152{tag}_gan/G.pth'
    dst_dataset_path = (
        f'./dataset/lokt_ffhq64_facescrub64_ir152{tag}_dataset/dataset.pt'
    )

    batch_size = 200
    device_ids_str = '5'

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    # dataset generator

    z_dim = 128

    generator = auto_generator_from_pretrained(generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
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

    dataset.save(dst_dataset_path)


for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    #     main(tag, 6)

    # # main(tags[1])

    # for tag in tags:
    try:
        main(tag)
    except Exception as e:
        pass
