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

from ...utils import (
    ClassificationLoss,
    BaseConstraint,
    DictAccumulator,
    batch_apply,
    unwrapped_parallel_module,
)
from ...models import BaseImageClassifier, BaseImageGenerator, C2fOutputMapping
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
        self.populations = torch.cat(
            [self.populations[elite_idx].unsqueeze(0), children], dim=0
        )

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

    final_num: int = 5


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

            result_labels.append(target_labels[: config.final_num])
            agent._compute_scores()

            _, use_indices = torch.topk(agent.scores, config.final_num)

            result_latents.append(agent.populations[use_indices])

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


@dataclass
class GeneticAlgorithm_c2f:

    batch_size: int
    populations: Tensor  # (num, 1, wdim)
    labels: LongTensor
    # noise_probability: int
    # noise_apply_fn: Callable[[Tensor, Tensor], Tensor]
    latent_constraint: BaseConstraint
    latent_score_fn: Callable[[Tensor, LongTensor], Tensor]
    temporal_trunctation_scheduler: Callable[[int, int], float] = (
        lambda step, max_step: 0.5
    )
    spatio_trunctation_scheduler: Callable[[int, int], float] = (
        lambda step, max_step: 0.02
    )
    crossover_beta_scheduler: Callable[[int, int], float] = lambda step, max_step: (
        0.1 if step / max_step < 0.94 else 0.1 * (max_step - step) / (0.16 * max_step)
    )
    scores = None  # (population num, )
    _step = 0
    _max_step = 300

    @torch.no_grad()
    def _compute_scores(self, latents):
        scores = batch_apply(
            self.latent_score_fn,
            latents,
            self.labels,
            batch_size=self.batch_size,
        )
        scores = torch.softmax(scores, dim=-1)
        return scores

    @torch.no_grad()
    def _find_elite(self):
        elite_idx = torch.argmax(self.scores, dim=0)
        return elite_idx.item()

    @torch.no_grad()
    def _get_parents(self, k):
        # weights = self.scores.tolist()
        parents_idx = random.choices(list(range(len(self.populations))), k=2 * k)
        return parents_idx[:k], parents_idx[k:]

    @torch.no_grad()
    def _cross_over(self, raw_child, parents1_idx, parents2_idx):
        beta = self.crossover_beta_scheduler(self._step, self._max_step)

        return raw_child + beta * (
            self.populations[parents1_idx] - self.populations[parents2_idx]
        )

    # @torch.no_grad()
    # def _mutate(self, children):
    #     mask = torch.rand_like(children) < self.noise_probability
    #     children = self.noise_apply_fn(children, mask)
    #     return children

    @torch.no_grad()
    def _get_raw_children(self, elite):
        trunctation = self.spatio_trunctation_scheduler(self._step, self._max_step)
        children = self.populations + trunctation * (elite - self.populations)

        return children

    @torch.no_grad()
    def temporal_trunctate(self, children):
        trunctation = self.temporal_trunctation_scheduler(self._step, self._max_step)
        mask = torch.rand_like(children) < trunctation
        return torch.where(mask, self.populations, children)

    @torch.no_grad()
    def step(self, step, max_step):
        self._step = step
        self._max_step = max_step
        self.scores = self._compute_scores(self.populations)
        # print(self.scores.shape, self.populations.shape)
        # exit()
        elite_idx = self._find_elite()
        elite = self.populations[elite_idx].unsqueeze(0)

        raw_children = self._get_raw_children(elite)

        parents1, parents2 = self._get_parents(len(self.populations))

        cross_over_children = self._cross_over(raw_children, parents1, parents2)

        children = self.temporal_trunctate(cross_over_children)

        if self.latent_constraint is not None:
            children = self.latent_constraint(children)

        children_scores = self._compute_scores(children)

        # print(
        #     children_scores.device,
        #     self.scores.device,
        #     children.device,
        #     self.populations.device,
        # )

        # exit()
        self.populations = torch.where(
            (children_scores > self.scores).reshape(-1, *([1] * (children.ndim - 1))),
            children,
            self.populations,
        )


@dataclass
class C2fGeneticOptimizationConfig(BaseImageOptimizationConfig):

    batch_size: int = 25

    first_optim_times: int = 20
    first_optim_select_num: int = 10
    population_num: int = 32

    genetic_iter_times: int = 300

    # noise_probability: int = 0.1
    latent_constraint: BaseConstraint = None
    # noise_apply_fn: Callable[[Tensor, Tensor], Tensor] = default_noise_apply_fn

    final_num: int = 5


class C2fGeneticOptimization(BaseImageOptimization):

    def __init__(
        self,
        config: C2fGeneticOptimizationConfig,
        generator: BaseImageGenerator,
        image_score_fn: Callable[
            [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        ],
        embed_module: BaseImageClassifier,
        mapping_module: C2fOutputMapping,
        gan_to_embeded_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(config)

        self.config: C2fGeneticOptimizationConfig
        self.generator = generator
        self.image_score_fn = image_score_fn
        self.embed_module = embed_module
        self.gan_to_embeded_transform = gan_to_embeded_transform
        self.mapping_module = mapping_module

    @torch.no_grad()
    def _latent_score_fn(self, latents: Tensor, labels: LongTensor):
        latents = latents.to(self.config.device)
        labels = labels.to(self.config.device)
        images = self.generator(latents, labels=labels)
        score = self.image_score_fn(images, labels)

        if isinstance(score, Tuple):
            score = score[0]
        # print(images.shape, score.shape, type(self.image_score_fn))
        # exit()
        return (score).detach().cpu()

    # @torch.no_grad()
    def _gen_images(self, latents: Tensor, labels: LongTensor):
        latents = latents.to(self.config.device)
        labels = labels.to(self.config.device)
        images = self.generator(latents, labels=labels)
        return images

    def _first_step(self, latents: Tensor, target: int):
        device = self.config.device
        latents = latents.to(device)
        labels = (torch.ones((len(latents),), dtype=torch.long) * target).to(device)
        latents = latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=0.02)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.config.first_optim_times, gamma=0.95
        )

        for i in tqdm(range(self.config.first_optim_times), leave=False):
            images = self._gen_images(latents, labels)
            # print(images.device)
            if self.gan_to_embeded_transform is not None:
                images = self.gan_to_embeded_transform(images)
            # print(images.device)
            # print(next(self.embed_module.parameters()).device)
            # exit()
            embed_result = self.embed_module(images)[0]

            prediction = torch.abs(
                torch.randn(
                    (
                        len(latents),
                        unwrapped_parallel_module(self.mapping_module).input_dim,
                    )
                )
            ).to(device)
            prediction[:, target] = 1e18
            prediction = F.normalize(prediction, dim=1)
            inverse_feature = self.mapping_module(prediction)
            loss = F.mse_loss(embed_result, inverse_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        latents = latents.detach()
        latents.requires_grad_(False)
        if len(latents) < self.config.first_optim_select_num:
            return latents.cpu()

        scores = self._latent_score_fn(latents, labels)
        assert scores.ndim == 1
        _, indices = torch.topk(scores, k=self.config.first_optim_select_num)
        return latents[indices].cpu()

    def _extend_latents(self, latents):
        num_lack = self.config.population_num - len(latents)
        if num_lack <= 0:
            return latents[: self.config.population_num]

        add_indices = torch.randint(0, len(latents), (3, num_lack))
        add_latents = (
            latents[add_indices]
            * torch.tensor([0.5, 0.25, 0.25])
            .reshape(3, *([1] * latents.ndim))
            .to(latents.device)
        ).sum(dim=0)

        add_latents = 0.7 * add_latents + 0.3 * torch.randn_like(add_latents)

        return torch.cat([latents, add_latents], dim=0)

    # @torch.no_grad()
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

            target_latents = self._first_step(target_latents, target)
            target_latents = self._extend_latents(target_latents)

            target_labels = (
                torch.ones((len(target_latents),), dtype=torch.long) * target
            )

            agent = GeneticAlgorithm_c2f(
                config.batch_size,
                target_latents,
                target_labels,
                # config.noise_probability,
                # config.noise_apply_fn,
                latent_constraint=config.latent_constraint,
                latent_score_fn=self._latent_score_fn,
            )

            for i in tqdm(range(config.genetic_iter_times)):
                agent.step(i, config.genetic_iter_times)

            result_labels.append(target_labels[: config.final_num])
            agent._compute_scores(agent.populations)

            _, use_indices = torch.topk(agent.scores, config.final_num)

            result_latents.append(agent.populations[use_indices])

        result_labels = torch.cat(result_labels, dim=0)
        result_latents = torch.cat(result_latents, dim=0)

        with torch.no_grad():
            result_images = batch_apply(
                self._gen_images,
                result_latents,
                result_labels,
                batch_size=config.batch_size,
                description=f'generate optimized images',
            )

        # return result_images, result_labels.detach().cpu()
        return ImageOptimizationOutput(
            images=result_images.detach().cpu(),
            labels=result_labels.detach().cpu(),
            latents=latents.detach().cpu(),
        )
