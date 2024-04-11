from abc import abstractmethod
from collections import OrderedDict
from typing import Callable, Any, Optional, Iterable, Sequence


import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from ..utils import TorchLoss, reparameterize
from ..models import (
    BaseImageClassifier,
    GmiDiscriminator64,
    GmiDiscriminator256,
    KedmiDiscriminator64,
    KedmiDiscriminator256,
    HOOK_NAME_FEATURE,
)


class BaseImageLoss(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, images: Tensor, labels: LongTensor, *args, **kwargs):
        pass


class ImageAugmentClassificationLoss(BaseImageLoss):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        loss_fn: str | Callable[[Tensor, LongTensor], Tensor] = 'cross_entropy',
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if create_aug_images_fn is None:
            create_aug_images_fn = lambda x: [x]

        self.create_aug_images_fn = create_aug_images_fn
        self.classifier = classifier
        self.loss_fn = TorchLoss(loss_fn, *args, **kwargs)

    def forward(self, images, labels, *args, **kwargs):
        acc = 0
        loss = 0
        total_num = 0
        for aug_images in self.create_aug_images_fn(images):
            total_num += 1
            conf, _ = self.classifier(aug_images)
            pred_labels = torch.argmax(conf, dim=-1)
            loss += self.loss_fn(conf, labels)
            # print(pred_labels)
            # print(labels)
            # exit()
            acc += (pred_labels == labels).float().mean().item()

        return loss, OrderedDict(
            [['classification loss', loss.item()], ['target acc', acc / total_num]]
        )


class ClassificationWithFeatureDistributionLoss(ImageAugmentClassificationLoss):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        feature_mean: Tensor,
        feature_std: Tensor,
        classification_loss_fn: (
            str | Callable[[Tensor, LongTensor], Tensor]
        ) = 'cross_entropy',
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        feature_loss_weight: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            classifier, classification_loss_fn, create_aug_images_fn, *args, **kwargs
        )

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_loss_weight = feature_loss_weight

    def _sample_distribution(self):
        return reparameterize(self.feature_mean, self.feature_std)

    def forward(self, images, labels, *args, **kwargs):

        acc = 0
        iden_loss = 0
        feature_loss = 0
        total_num = 0
        bs = len(images)
        for aug_images in self.create_aug_images_fn(images):
            total_num += 1
            conf, info_dict = self.classifier(aug_images)
            if HOOK_NAME_FEATURE not in info_dict:
                raise RuntimeError(
                    f'The addition info that the model outputs do not contains {HOOK_NAME_FEATURE}'
                )
            pred_labels = torch.argmax(conf, dim=-1)
            iden_loss += self.loss_fn(conf, labels)

            feature_dist_samples = self._sample_distribution()
            feature_loss += torch.mean(
                (
                    info_dict[HOOK_NAME_FEATURE].view(bs, -1)
                    - feature_dist_samples.view(1, -1)
                ).pow(2)
            )
            acc += (pred_labels == labels).float().mean().item()

        loss = iden_loss + self.feature_loss_weight * feature_loss

        return loss, OrderedDict(
            [
                ['loss', loss.item()],
                ['classification loss', iden_loss.item()],
                ['feature loss', feature_loss.item()],
                ['target acc', acc / total_num],
            ]
        )


class GmiDiscriminatorLoss(BaseImageLoss):

    def __init__(
        self, discriminator: GmiDiscriminator64 | GmiDiscriminator256, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator

    def forward(self, images, labels, *args, **kwargs):
        dis_res = self.discriminator(images)
        loss = -dis_res.mean()
        return loss, {'discriminator loss': loss.item()}


class KedmiDiscriminatorLoss(BaseImageLoss):

    def __init__(
        self,
        discriminator: KedmiDiscriminator64 | KedmiDiscriminator256,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator

    def forward(self, images, labels, *args, **kwargs):
        _, dis_res = self.discriminator(images)
        logsumup = torch.logsumexp(dis_res, dim=-1)
        # loss = - dis_res.mean()
        loss = torch.mean(F.softplus(logsumup)) - torch.mean(logsumup)
        return loss, {'discriminator loss': loss.item()}


class ComposeImageLoss(BaseImageLoss):

    def __init__(
        self,
        losses: list[BaseImageLoss],
        weights: Optional[list[float]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if weights is None:
            weights = [1] * len(losses)

        if len(losses) == 0:
            raise RuntimeError(f'losses should be at least one function.')

        if len(weights) != len(losses):
            raise RuntimeError(
                f'Expect the equal length of losses and weights, but found the fronter {len(losses)} and the latter {len(weights)}.'
            )

        self.losses = losses
        self.weights = weights

    def forward(self, images: Tensor, labels: LongTensor, *args, **kwargs):

        compose_loss = 0.0
        return_dict = OrderedDict()
        return_dict['compose loss'] = 0

        for lossfn, weight in zip(self.losses, self.weights):

            loss = lossfn(images, labels, *args, **kwargs)
            if not isinstance(loss, Tensor):
                loss, single_dict = loss
                for k, v in single_dict.items():
                    return_dict[k] = v
            compose_loss += weight * loss

        return_dict['compose loss'] = compose_loss.item()

        return compose_loss, return_dict
