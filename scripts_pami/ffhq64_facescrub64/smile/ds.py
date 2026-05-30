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
    get_stylegan2ada_generator,
)
from modelinversion.datasets import (
    generator_generate_datasets,
    preprocess_facescrub_fn,
    GeneratorDataset,
)


def main(tag='no'):

    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    tag = '_' + tag

    num_classes = 100
    stylegan2ada_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/stylegan2-ffhq-256x256.pkl'

    dst_dataset_path = f'./dataset/plg_ffhq64_facescrub64_ir152_{tag}_dataset'

    batch_size = 100
    device_ids_str = '5'

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    # dataset generator

    z_dim = 512

    _, generator = get_stylegan2ada_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, single_w=True, optimize_in_w=False
    )
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
    target_model.eval()

    gan_to_target_transform = TF.Compose(
        [TF.CenterCrop((176, 176)), TF.Resize((64, 64))]
    )

    dataset = GeneratorDataset.create(
        z_dim,
        num_classes=num_classes,
        generate_num_per_class=200,
        generator=generator,
        target_model=target_model,
        batch_size=batch_size,
        device=device,
        gan_to_target_transform=gan_to_target_transform,
    )

    dataset.save(dst_dataset_path)


if __name__ == '__main__':
    for tag in [
        'no',
        'vib0.01',
        'bido0.01_0.1_pretrain',
        'ls0.05',
        'tl0.5',
        'rolss0.0_2',
    ]:
        main(tag)
