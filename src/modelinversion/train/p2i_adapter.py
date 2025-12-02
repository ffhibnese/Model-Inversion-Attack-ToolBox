from tqdm import tqdm
from typing import Optional

import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from ..models.adapters.p2i import P2IConfidence2StyleAdapter
from ..models.classifiers import BaseImageClassifier
from ..models.gans import StyleGAN2adaSynthesisWrapper, StyleGan2adaMappingWrapper
from ..utils import unwrapped_parallel_module


def _get_first(data):
    if not isinstance(data, Tensor):
        return data[0]
    return data


def train_p2i_adapter(
    epoch_num: int,
    target_model: BaseImageClassifier,
    p2i_adapter: P2IConfidence2StyleAdapter,
    optimizer: Optimizer,
    # src_model: Module,
    # dst_model: Module,
    stylegan_mapping: StyleGan2adaMappingWrapper,
    stylegan: StyleGAN2adaSynthesisWrapper,
    avg_w: torch.Tensor,
    real_dataloader: DataLoader,
    fake_dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    lpips_loss_fn,
    arcface_loss_fn,
    landmark_loss_fn,
    # fake_loss_fns: list[tuple[str, nn.Module]],
    # real_loss_fns: list[tuple[str, nn.Module]],
    schedular: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    show_info_iters: int = 100,
):
    os.makedirs(save_path, exist_ok=True)
    torch.save(avg_w, os.path.join(save_path, "avg_w.pth"))
    for epoch in range(epoch_num):

        for real_data, fake_data in tqdm(
            zip(real_dataloader, fake_dataloader), total=len(real_dataloader)
        ):
            fake_z, fake_images, fake_logits = (
                fake_data['z'].to(device),
                fake_data['image'].to(device),
                fake_data['logits'].to(device),
            )
            # print(fake_logits.shape, "fake logit")
            # exit()
            fake_logp = torch.log_softmax(fake_logits, dim=-1)

            real_images = real_data[0]
            real_images = real_images.to(device)
            real_logits = target_model(real_images)
            real_logits = _get_first(real_logits)
            real_logp = torch.log_softmax(real_logits, dim=-1)

            fake_w = stylegan_mapping(fake_z)

            recon_real_w, _, recon_real_inimages = p2i_adapter(real_logp)
            recon_real_w = recon_real_w + avg_w
            recon_real_images = stylegan(recon_real_w)

            recon_fake_w, _, recon_fake_inimages = p2i_adapter(fake_logp)
            recon_fake_w = recon_fake_w + avg_w
            recon_fake_images = stylegan(recon_fake_w)

            # l2 loss
            l2_loss = F.mse_loss(recon_fake_images, fake_images) + F.mse_loss(
                recon_fake_inimages,
                F.interpolate(fake_images, size=recon_fake_images.shape[-2:]),
            )
            # lpips loss
            lpips_loss = lpips_loss_fn(recon_fake_images, fake_images) + lpips_loss_fn(
                recon_real_images, real_images
            )
            # arcface loss
            arcface_loss = arcface_loss_fn(
                recon_fake_images, fake_images
            ) + arcface_loss_fn(recon_real_images, real_images)
            # landmark loss
            landmark_loss = landmark_loss_fn(
                recon_fake_images, fake_images
            ) + landmark_loss_fn(recon_real_images, real_images)

            loss = l2_loss + lpips_loss + arcface_loss + landmark_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if schedular is not None:
                schedular.step()

        unwrapped_parallel_module(p2i_adapter).save_pretrained(
            os.path.join(save_path, f"p2i_adapter.pth")
        )
