import sys
import os
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop

from modelinversion.models import (
    SimpleGenerator64,
    GmiDiscriminator64,
    auto_classifier_from_pretrained,
    get_stylegan2ada_generator,
)
from modelinversion.models.adapters import P2IConfidence2StyleAdapter
from modelinversion.train import GmiGanTrainer, GmiGanTrainConfig, train_p2i_adapter
from modelinversion.utils import Logger
from modelinversion.utils.losses import LpipsLoss, ArcfaceIDCosineLoss, LandmarkLoss
from modelinversion.datasets import GeneratorDataset
from modelinversion.sampler import SimpleLatentsSampler


def main(tag):

    paths = get_attack_paths('p2i', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    num_classes = 530

    arcface_ckpt_path = (
        "/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/p2i/arcface.pth"
    )
    parsing_model_ckpt_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/p2i/parsing_model.pth'
    )

    batch_size = 8

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.p2i_adapter_path, f'train_gan_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model.eval()

    mapping, generator = get_stylegan2ada_generator(
        paths.stylegan2ada_path, paths.stylegan2ada_ckpt_path, optimize_in_w=True
    )

    mapping = mapping.to(device).eval()
    generator = generator.to(device).eval()

    lpips_loss_fn = LpipsLoss(device=device, scale=1.0)
    arcface_loss_fn = ArcfaceIDCosineLoss(
        ckpt_path=arcface_ckpt_path, device=device, scale=1
    )

    landmark_loss_fn = LandmarkLoss(
        ckpt_path=parsing_model_ckpt_path, device=device, scale=1
    )

    # prepare dataset

    from torchvision.datasets import ImageFolder

    real_dataset = ImageFolder(
        paths.p2i_real_dataset_path,
        transform=Compose(
            [
                Resize((256, 256)),
                ToTensor(),
            ]
        ),
    )
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    fake_dataset = GeneratorDataset.from_precreate(
        paths.p2i_fake_dataset_path,
        generator=get_stylegan2ada_generator(
            paths.stylegan2ada_path, paths.stylegan2ada_ckpt_path, optimize_in_w=False
        )[1]
        .to(device)
        .eval(),
        device=device,
        transform=Compose(
            [
                CenterCrop(176),
                Resize((256, 256)),
            ]
        ),
    )

    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=fake_dataset.collate_fn_for_p2i,
    )

    p2i_adapter = P2IConfidence2StyleAdapter(
        num_classes=num_classes,
        n_styles=14,
        arcface_model_path=arcface_ckpt_path,
    )
    p2i_adapter = p2i_adapter.to(device).train()

    # prepare min max constrain

    with torch.no_grad():
        simple_sampler = SimpleLatentsSampler(
            input_size=512, batch_size=batch_size, latents_mapping=mapping
        )

        all_w = simple_sampler([0], 5000)[0]
        all_w_means = torch.mean(all_w, dim=0, keepdim=True).to(device)

    optimizer = torch.optim.Adam(p2i_adapter.parameters(), lr=0.001)
    optim_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.99
    )

    train_p2i_adapter(
        40,
        target_model=target_model,
        p2i_adapter=p2i_adapter,
        optimizer=optimizer,
        stylegan_mapping=mapping,
        stylegan=generator,
        avg_w=all_w_means,
        real_dataloader=real_dataloader,
        fake_dataloader=fake_dataloader,
        device=device,
        save_path=paths.p2i_adapter_path,
        lpips_loss_fn=lpips_loss_fn,
        arcface_loss_fn=arcface_loss_fn,
        landmark_loss_fn=landmark_loss_fn,
        schedular=optim_scheduler,
    )

    logger.close()


for tag in ALL_TAGS:
    main(tag)
