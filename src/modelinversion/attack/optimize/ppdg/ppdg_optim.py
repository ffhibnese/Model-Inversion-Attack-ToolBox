import os
import copy
import importlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch.optim import Optimizer, Adam
from tqdm import tqdm


from ..base import (
    ImageAugmentWhiteBoxOptimization,
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageOptimizationOutput,
)
from ....utils import (
    ClassificationLoss,
    BaseConstraint,
    DictAccumulator,
    obj_to_yaml,
    BaseOutput,
    freeze,
    unfreeze,
)
from ....models import BaseImageClassifier, BaseImageGenerator
from ....scores import BaseLatentScore
from ....sampler import BaseLatentsSampler
from .mmd import mmd


@dataclass
class PPDGWhiteBoxOptimizationConfig(ImageAugmentWhiteBoxOptimizationConfig):

    # tuning_type: Literal['mmd'] = 'mmd'
    tuning_image_num: int = 5
    tuning_discriminator: nn.Module = None
    tuning_discriminator_ratio: float = 0.1
    tuning_steps: int = 150
    w_mean: torch.Tensor = None
    w_std: torch.Tensor = None
    tuning_feature_extractor_path: str = (
        'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    )
    tuning_lr: float = 0.0003

    inversion_steps: int = 300
    inversion_lr: float = 0.01
    inversion_noise_ratio: str = 0.05
    inversion_lr_rampdown_length: float = 0.25
    inversion_lr_rampup_length: float = 0.05
    inversion_noise_ramp_length: float = 0.75

    # optimizer: str | type = 'Adam'
    # optimizer_kwargs: dict = field(default_factory=lambda: {})
    # iter_times: int = 600
    # show_loss_info_iters: int = 100

    # latent_constraint: Optional[BaseConstraint] = None


class PPDGWhiteBoxOptimization(ImageAugmentWhiteBoxOptimization):
    """Base class for all white-box optimization classes.

    Args:
        config (SimpleWhiteBoxOptimizationConfig):
            Config of the white box optimization.
        generator (BaseImageGenerator):
            Generator to generate images from latent vectors.
        image_loss_fn (Callable[[Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]]):
            A function to calculate loss of the generated images with given labels. Returns the loss and an optional OrderedDict that contains loss information to show.
    """

    def __init__(
        self,
        config: PPDGWhiteBoxOptimizationConfig,
        # generator: BaseImageGenerator,
        # image_loss_fn: Callable[
        #     [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        # ],
        generator: BaseImageGenerator,
        target_model: BaseImageClassifier,
    ) -> None:
        super().__init__(config, generator, target_model)

        # optimizer_class = config.optimizer
        # if isinstance(optimizer_class, str):
        #     optim_module = importlib.import_module('torch.optim')
        #     optimizer_class = getattr(optim_module, optimizer_class, None)

        # if not hasattr(optimizer_class, 'zero_grad'):
        #     raise RuntimeError('Optimizer do not has attribute `zero_grad`')

        # if not hasattr(optimizer_class, 'step'):
        #     raise RuntimeError('Optimizer do not has attribute `step`')

        # self.optimizer_class = optimizer_class
        # self.generator = generator
        # self.image_loss_fn = image_loss_fn
        pass

    def prepare_generator(
        self, generator, attack_result: ImageOptimizationOutput
    ) -> None:
        target_images = attack_result.images.to(device=self.config.device)
        new_generator = copy.deepcopy(generator)

        unfreeze(new_generator)

        config: PPDGWhiteBoxOptimizationConfig = self.config
        device = config.device

        if 'https://' in config.tuning_feature_extractor_path:
            os.system("wget -O vgg16.pt")
            config.tuning_feature_extractor_path = 'vgg16.pt'

        feature_extractor = (
            torch.jit.load(config.tuning_feature_extractor_path).eval().to(device)
        )

        target_images = (target_images.to(device) + 1) * 255 / 2
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        # target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = (
            feature_extractor(target_images, resize_images=False, return_lpips=True)
            .detach()
            .requires_grad_(False)
        )

        ws = self.prepare_inversion_w(
            new_generator,
            attack_result,
            target_images,
            target_features,
            feature_extractor,
        )

        optimizer = torch.optim.Adam(new_generator.parameters(), lr=config.tuning_lr)
        new_generator.train()

        for step in tqdm(range(config.tuning_steps), leave=False):
            generate_images = new_generator(ws, labels=attack_result.labels.to(device))
            generate_images = (generate_images.to(device) + 1) * 255 / 2
            generate_images = F.interpolate(
                generate_images, size=(256, 256), mode='area'
            )
            generate_features = feature_extractor(
                generate_images, resize_images=False, return_lpips=True
            )

            # 2 - MMD loss (synth_features - target features)
            loss = mmd(generate_features, target_features)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        freeze(new_generator)

        new_generator.eval()

        return new_generator

    def prepare_inversion_w(
        self,
        generator,
        attack_result: ImageOptimizationOutput,
        target_images,
        target_features,
        feature_extractor,
    ) -> Tensor:

        config: PPDGWhiteBoxOptimizationConfig = self.config
        device = config.device

        generator.eval()

        start_w = (
            torch.tile(config.w_mean, (len(target_images), 1, 1))
            .clone()
            .detach()
            .to(device)
            .requires_grad_(True)
        )
        w_opt = start_w
        # w_opt = torch.tensor(
        #     start_w, dtype=torch.float32, device=device, requires_grad=True
        # )
        optimizer = torch.optim.Adam(
            [w_opt], betas=(0.9, 0.999), lr=config.inversion_lr
        )

        for step in tqdm(range(config.inversion_steps), leave=False):

            t = step / config.inversion_steps
            lr_ramp = min(1.0, (1.0 - t) / config.inversion_lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / config.inversion_lr_rampup_length)
            lr = config.inversion_lr * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            w_noise_scale = (
                config.w_std.detach()
                * config.inversion_noise_ratio
                * max(0.0, 1.0 - t / config.inversion_noise_ramp_length) ** 2
            )
            w_noise = torch.randn_like(w_opt) * w_noise_scale

            ws = w_opt + w_noise

            generate_images = generator(ws, labels=attack_result.labels.to(device))

            # 1 - Discriminator loss
            # print(generate_images.shape)
            # exit()
            discriminator_logits = config.tuning_discriminator(generate_images, None)
            discriminator_loss = nn.functional.softplus(-discriminator_logits).mean()

            generate_images = (generate_images.to(device) + 1) * 255 / 2
            generate_images = F.interpolate(
                generate_images, size=(256, 256), mode='area'
            )
            generate_features = feature_extractor(
                generate_images, resize_images=False, return_lpips=True
            )

            # 2 - MMD loss (synth_features - target features)
            MMD_loss = mmd(generate_features, target_features)
            # MMD_loss = 0

            loss = MMD_loss + config.tuning_discriminator_ratio * discriminator_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return w_opt

    def attack_one_round(self, latents: Tensor, labels: LongTensor, generator):
        ori_generator = self.generator
        self.generator = generator
        result = super().__call__(latents, labels)
        self.generator = ori_generator
        return result

    def __call__(
        self, latents: Tensor, labels: LongTensor
    ) -> Tuple[Tensor, LongTensor]:

        first_attack_result = self.attack_one_round(latents, labels, self.generator)
        first_attack_result.images = first_attack_result.images[
            : self.config.tuning_image_num
        ]
        first_attack_result.labels = first_attack_result.labels[
            : self.config.tuning_image_num
        ]
        first_attack_result.latents = first_attack_result.latents[
            : self.config.tuning_image_num
        ]
        generator = self.prepare_generator(self.generator, first_attack_result)
        return self.attack_one_round(latents, labels, generator)
