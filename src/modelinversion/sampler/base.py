import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable, Sequence

import torch
from torch import Tensor, LongTensor

from ..models import (
    BaseImageGenerator,
    BaseImageClassifier,
    StyleGAN2adaSynthesisWrapper,
)
from .flow import MixtureOfGMM, LayeredMineGAN, LayeredFlowMiner, FlowConfig
from ..utils import batch_apply


class BaseLatentsSampler(ABC):
    """The base class for latent vectors samplers."""

    @abstractmethod
    def __call__(self, labels: list[int], sample_num: int):
        """The sampling function of the sampler.

        Args:
            labels (list[int]): The labels where latent vectors are sampled.
            sample_num (int): The number of latent vectors sampled.
        """
        pass


class SimpleLatentsSampler(BaseLatentsSampler):
    """A latent vector sampler that generates Gaussian distributed random latent vectors with the given `input_size`.

    Args:
        input_size (int or Sequence[int]): The shape of the latent vectors.
    """

    def __init__(
        self,
        input_size: int | Sequence[int],
        batch_size: int,
        latents_mapping: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if isinstance(input_size, int):
            input_size = (input_size,)
        if not isinstance(input_size, tuple):
            input_size = tuple(input_size)
        self._input_size = input_size
        self.batch_size = batch_size
        self.latents_mapping = latents_mapping

    def get_batch_latent_size(self, batch_num: int):
        return (batch_num,) + self._input_size

    def __call__(self, labels: list[int], sample_num: int):
        size = self.get_batch_latent_size(sample_num)

        latents = torch.randn(size)
        if self.latents_mapping is not None:
            latents = batch_apply(
                self.latents_mapping, latents, batch_size=self.batch_size
            )
        return {label: latents.detach().clone() for label in labels}


@torch.no_grad()
def cross_image_augment_scores(
    model: BaseImageClassifier,
    device: torch.device,
    create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
    images: Tensor,
):
    images = images.detach().to(device)

    if create_aug_images_fn is not None:
        scores = 0
        total_num = 0
        for trans in create_aug_images_fn(images):
            total_num += 1
            conf = model(trans)[0].softmax(dim=-1).cpu()
            scores += conf
        res = scores / total_num
    else:
        conf = model(images)[0].softmax(dim=-1).cpu()
        res = conf

    return res


class ImageAugmentSelectLatentsSampler(SimpleLatentsSampler):

    def __init__(
        self,
        input_size: int | Sequence[int],
        batch_size: int,
        all_sample_num: int,
        generator: BaseImageGenerator,
        classifier: BaseImageClassifier,
        device: torch.device,
        latents_mapping: Optional[Callable] = None,
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
    ) -> None:
        super().__init__(input_size, batch_size, latents_mapping)

        self.all_sample_num = all_sample_num
        self.generator = generator
        self.classifier = classifier
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn

    def _get_score(self, latents: Tensor, labels: list[int]):

        images = self.generator(latents.to(self.device))
        return cross_image_augment_scores(
            self.classifier, self.device, self.create_aug_images_fn, images
        )

    def __call__(self, labels: list[int], sample_num: int):
        all_sample_num = self.all_sample_num
        if sample_num > self.all_sample_num:
            all_sample_num = sample_num

        raw_size = self.get_batch_latent_size(all_sample_num)
        latents = torch.randn(raw_size)
        if self.latents_mapping is not None:
            latents = batch_apply(
                self.latents_mapping, latents, batch_size=self.batch_size
            )
        scores = batch_apply(
            self._get_score,
            latents,
            batch_size=self.batch_size,
            labels=labels,
            use_tqdm=True,
        )

        results = {}
        for label in labels:
            scores_label = scores[:, label]
            _, indices = torch.topk(scores_label, k=sample_num)
            results[label] = latents[indices].detach().cpu()
        return results


class GaussianMixtureLatentsSampler(SimpleLatentsSampler):

    def __init__(
        self,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        k: int,
        n_components: int,
        l: int,
        l_identity: list,
        device: torch.device,
        latents_mapping: Optional[Callable] = None,
        mode: str = 'eval',
        **kwargs
    ) -> None:
        """
        input
                k: num dim
                l: num component
        """
        super().__init__(input_size, batch_size, latents_mapping)
        assert latents_mapping != None
        self.mode = mode
        self.miner = MixtureOfGMM(k, n_components, l)
        if mode == 'eval':
            self.miner.load_state_dict(torch.load(kwargs['path'], map_location='cpu'))
            self.miner.eval()
        self.miner = self.miner.to(device).double()
        self.minegan_Gmapping = LayeredMineGAN(self.miner, self.latents_mapping)
        self.l_identity = l_identity
        self.device = device
        self.generator = generator

        self.identity_mask = torch.zeros(1, l, 1).to(device)
        self.identity_mask[:, self.l_identity, :] = 1

    def __call__(self, label: int, sample_num: int):
        size = self.get_batch_latent_size(sample_num)

        results = {}
        z_nuisance = torch.randn(size).to(self.device).double()
        z_identity = torch.randn(size).to(self.device).double()
        w_nuisance = batch_apply(
            self.latents_mapping, z_nuisance, batch_size=self.batch_size
        )
        w_identity = batch_apply(
            self.minegan_Gmapping, z_identity, batch_size=self.batch_size
        )
        w = (1 - self.identity_mask) * w_nuisance + self.identity_mask * w_identity
        results[label] = w.detach().clone()
        return results


class LayeredFlowLatentsSampler(SimpleLatentsSampler):

    def __init__(
        self,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        flow_params: FlowConfig,
        device: torch.device,
        latents_mapping: Optional[Callable],
        mode: str = 'eval',
        **kwargs,
    ) -> None:
        """
        input
                k: num dim
                l: num component
        """
        super().__init__(input_size, batch_size, latents_mapping)
        assert latents_mapping != None
        self.miner = (
            LayeredFlowMiner(
                flow_params.k,
                flow_params.l,
                flow_params.flow_permutation,
                flow_params.flow_K,
                flow_params.flow_glow,
                flow_params.flow_coupling,
                flow_params.flow_L,
                flow_params.flow_use_actnorm,
            )
        )
        if mode == 'eval':
            self.miner.load_state_dict(torch.load(kwargs['path'], map_location='cpu'))
            self.miner.eval()
        elif mode == 'train':
            self.miner.train()
        self.miner = self.miner.to(device).double()
        self.minegan_Gmapping = LayeredMineGAN(self.miner, self.latents_mapping)
        self.l_identity = flow_params.l_identity
        self.device = device
        self.generator = generator

        self.identity_mask = torch.zeros(1, flow_params.l, 1).to(device)
        self.identity_mask[:, self.l_identity, :] = 1

    def __call__(self, label: int, sample_num: int):
        size = self.get_batch_latent_size(sample_num)

        results = {}
        z_nuisance = torch.randn(size).to(self.device).double()
        z_identity = torch.randn(size).to(self.device).double()
        w_nuisance = batch_apply(
            self.latents_mapping, z_nuisance, batch_size=self.batch_size
        )
        w_identity = batch_apply(
            self.minegan_Gmapping, z_identity, batch_size=self.batch_size
        )
        w = (1 - self.identity_mask) * w_nuisance + self.identity_mask * w_identity
        results[label] = w
        return results
