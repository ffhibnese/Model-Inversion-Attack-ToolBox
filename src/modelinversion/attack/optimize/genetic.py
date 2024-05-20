import os
import importlib
import random
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter, deque
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Iterable, Sequence
from functools import reduce


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor, LongTensor
from torch.optim import Optimizer, Adam
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm

from ...utils import ClassificationLoss, BaseConstraint, DictAccumulator, batch_apply
from ...models import BaseImageClassifier, BaseImageGenerator
from ...scores import BaseLatentScore
from .base import (
    BaseImageOptimizationConfig,
    BaseImageOptimization,
    ImageOptimizationOutput,
)


@dataclass
class GeneticAlgorithm:

    batch_size: int
    populations: Tensor  # (num, 1, wdim)
    labels: LongTensor
    noise_probability: int
    noise_apply_fn: Callable[[Tensor, Tensor], Tensor]
    latent_constraint: BaseConstraint
    latent_score_fn: Callable[[Tensor, LongTensor], Tensor]
    scores = None  # (population num, )

    @torch.no_grad()
    def _compute_scores(self):
        scores = batch_apply(
            self.latent_score_fn,
            self.populations,
            self.labels,
            batch_size=self.batch_size,
        )
        self.scores = torch.softmax(scores, dim=-1)

    @torch.no_grad()
    def _find_elite(self):
        elite_idx = torch.argmax(self.scores, dim=0)
        return elite_idx.item()

    @torch.no_grad()
    def _get_parents(self, k):
        weights = self.scores.tolist()
        parents_idx = random.choices(
            list(range(len(weights))), weights=weights, k=2 * k
        )
        return parents_idx[:k], parents_idx[k:]

    @torch.no_grad()
    def _cross_over(self, parents1_idx, parents2_idx):
        parents1_fitness_scores = self.scores[parents1_idx]
        parents2_fitness_scores = self.scores[parents2_idx]

        parents1 = self.populations[parents1_idx]  # size: N, 1, wdim
        parents2 = self.populations[parents2_idx]  # size: N, 1, wdim

        p = parents1_fitness_scores / (
            parents1_fitness_scores + parents2_fitness_scores
        )

        # print(parents1_fitness_scores.shape, parents2_fitness_scores.shape, p.shape)
        p = p.reshape(-1, *([1] * (len(parents1.shape) - 1)))

        # print(p.shape, parents1.shape, self.scores.shape)
        # exit()
        mask = torch.rand_like(parents1) < p
        return torch.where(mask, parents1, parents2)

    @torch.no_grad()
    def _mutate(self, children):
        mask = torch.rand_like(children) < self.noise_probability
        children = self.noise_apply_fn(children, mask)
        return children

    @torch.no_grad()
    def step(self):
        self._compute_scores()
        parents = self._get_parents(len(self.populations) - 1)
        children = self._cross_over(*parents)
        children = self._mutate(children)

        elite_idx = self._find_elite()
        # print(self.populations.shape)
        # print(self.populations[elite_idx].unsqueeze(0).shape)
        # print(children.shape)
        self.populations = torch.cat(
            [self.populations[elite_idx].unsqueeze(0), children], dim=0
        )
        # print(self.populations.shape)
        # exit()

        if self.latent_constraint is not None:
            self.populations = self.latent_constraint(self.populations)


def default_noise_apply_fn(latents, mask):
    return latents + mask * torch.randn_like(latents)


@dataclass
class GeneticOptimizationConfig(BaseImageOptimizationConfig):

    iter_times: int = 100
    batch_size: int = 25

    noise_probability: int = 0.1
    latent_constraint: BaseConstraint = None
    noise_apply_fn: Callable[[Tensor, Tensor], Tensor] = default_noise_apply_fn


class GeneticOptimization(BaseImageOptimization):

    def __init__(
        self,
        config: GeneticOptimizationConfig,
        generator: BaseImageGenerator,
        image_score_fn: Callable[
            [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        ],
    ) -> None:
        super().__init__(config)

        self.config: GeneticOptimizationConfig
        self.generator = generator
        self.image_score_fn = image_score_fn

    @torch.no_grad()
    def _latent_score_fn(self, latents: Tensor, labels: LongTensor):
        latents = latents.to(self.config.device)
        labels = labels.to(self.config.device)
        images = self.generator(latents, labels=labels)
        score = self.image_score_fn(images, labels)
        if isinstance(score, Tuple):
            score = score[0]
        return (score).detach().cpu()

    @torch.no_grad()
    def _gen_images(self, latents: Tensor, labels: LongTensor):
        latents = latents.to(self.config.device)
        labels = labels.to(self.config.device)
        images = self.generator(latents, labels=labels)
        return images.detach().cpu()

    @torch.no_grad()
    def __call__(
        self, latents: Tensor, labels: LongTensor
    ) -> Tuple[Tensor, LongTensor]:
        config = self.config
        targets = set(labels.tolist())

        result_labels, result_latents = [], []

        print(latents.shape)

        for target in targets:
            print(f'optimize class {target}')
            target_mask = labels == target

            target_latents = latents[target_mask]
            if len(target_mask) < 3:
                warnings.warn(
                    f'The number of latent vectors are too small. Found {len(target_mask)}'
                )
                continue

            target_labels = (
                torch.ones((len(target_latents),), dtype=torch.long) * target
            )
            agent = GeneticAlgorithm(
                config.batch_size,
                target_latents,
                target_labels,
                config.noise_probability,
                config.noise_apply_fn,
                latent_constraint=config.latent_constraint,
                latent_score_fn=self._latent_score_fn,
            )

            for _ in tqdm(range(config.iter_times)):
                agent.step()

            result_labels.append(target_labels)
            result_latents.append(agent.populations)

        result_labels = torch.cat(result_labels, dim=0)
        result_latents = torch.cat(result_latents, dim=0)

        result_images = batch_apply(
            self._gen_images,
            result_latents,
            result_labels,
            batch_size=config.batch_size,
            description=f'generate optimized images',
        )

        # return result_images, result_labels.detach().cpu()
        return ImageOptimizationOutput(
            images=result_images, labels=result_labels, latents=latents
        )
