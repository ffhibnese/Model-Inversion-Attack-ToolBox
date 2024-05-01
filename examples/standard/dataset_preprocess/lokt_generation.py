import sys
import os

sys.path.append('../../../src')

import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import (
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.datasets import (
    generator_generate_datasets,
    preprocess_celeba_fn,
    GeneratorDataset,
)

if __name__ == '__main__':

    num_classes = 1000
    generator_ckpt_path = (
        '../../../checkpoints_v2/attacks/lokt/lokt_celeba64_celeba64_ir152_G.pt'
    )
    target_model_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    )
    dst_dataset_path = '../../../results/lokt_celeba_celeba_ir152_dataset/celeba64_celeba64_ir152_dataset.pt'

    batch_size = 200
    device_ids_available = '3'

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # dataset generator

    z_dim = 128

    generator = auto_generator_from_pretrained(generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
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
