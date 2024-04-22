import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable

import torch
from torch import Tensor, LongTensor

from ..models import BaseImageClassifier
from ..utils import TorchLoss
from .functional import (
    specific_image_augment_scores,
    specific_image_augment_scores_label_only,
    specific_image_augment_loss_score,
)


class BaseImageClassificationScore(ABC):
    """This is a class for generating scores for each image with the corresponding label."""

    @abstractmethod
    def __call__(self, images: Tensor, labels: LongTensor) -> Tensor:
        """The scoring function to score all images with the corresponding labels.

        Args:
            images (Tensor): images.
            labels (LongTensor): The corresponding labels for images. The length of `labels` should keep the same as `images`

        Returns:
            Tensor: The score of each image.
        """
        pass


class ImageClassificationAugmentConfidence(BaseImageClassificationScore):
    """This is a class for generating score for each image with the corresponding label. The score is calculated by the conficence of the classifier model.

    Args:
        generator (BaseImageGenerator):
            The image generator.
        model (BaseImageClassifier):
            The image classifier to generate scores.
        device (device):
            The device used for calculation. Please keep the same with the device of `generator` and `model`.
        create_aug_images_fn (Callable[[Tensor], Iterable[Tensor]], optional):
            The function to create a list of augment images that will be used to calculate the score. Defaults to `None`.
    """

    def __init__(
        self,
        model: BaseImageClassifier,
        device: torch.device,
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: LongTensor) -> Tensor:

        return specific_image_augment_scores(
            self.model, self.device, self.create_aug_images_fn, images, labels
        )


class ImageClassificationAugmentLossScore(BaseImageClassificationScore):
    """This is a class for generating score for each image with the corresponding label with loss function. The score is calculated by the output of the classifier model and the loss function. The output is the inverse of the loss result.

    Args:
        generator (BaseImageGenerator):
            The image generator.
        model (BaseImageClassifier):
            The image classifier to generate scores.
        device (device):
            The device used for calculation. Please keep the same with the device of `generator` and `model`.
        create_aug_images_fn (Callable[[Tensor], Iterable[Tensor]], optional):
            The function to create a list of augment images that will be used to calculate the score. Defaults to `None`.
    """

    def __init__(
        self,
        model: BaseImageClassifier,
        device: torch.device,
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        loss_fn: Callable = 'ce',
    ) -> None:
        self.model = model
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn
        self.loss_fn = TorchLoss(loss_fn, reduction='none')

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: LongTensor) -> Tensor:

        return specific_image_augment_loss_score(
            self.model,
            self.device,
            self.create_aug_images_fn,
            images,
            labels,
            self.loss_fn,
        )


class ImageClassificationAugmentLabelOnlyScore(BaseImageClassificationScore):
    """This is a class for generating score for each image with the corresponding label. The score is `correct_score` if the image is classified correctly by the classifier, `wrong_score` otherwise.

    Args:
        classifier (BaseImageClassifier):
            The image classifier to generate scores.
        device (device):
            The device used for calculation. Please keep the same with the device of `classifier`.
        create_aug_images_fn (Callable[[Tensor], Iterable[Tensor]], optional):
            The function to create a list of augment images that will be used to calculate the score. Defaults to `None`.
        correct_score (int):
            The score when the image is classified correctly.
        wrong_score (int):
            The score when the image is misclassified.
    """

    def __init__(
        self,
        classifier: BaseImageClassifier,
        device: torch.device,
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        correct_score=1,
        wrong_score=-1,
    ) -> None:
        self.classifier = classifier
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn
        self.correct_score = correct_score
        self.wrong_score = wrong_score

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: LongTensor) -> Tensor:

        return specific_image_augment_scores_label_only(
            self.classifier,
            self.device,
            self.create_aug_images_fn,
            images,
            labels,
            self.correct_score,
            self.wrong_score,
        )


class ImageClassificationScoreCompose(BaseImageClassificationScore):
    """Compose of several `BaseImageClassificationScore`.

    Args:
        scores (list[BaseImageClassificationScore]): list of BaseImageClassificationScore to compose.
        weights (list[int], optional): weights of scores. Defaults to `None`.
    """

    def __init__(
        self,
        scores: list[BaseImageClassificationScore],
        weights: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.num = len(scores)
        self.scores = scores
        if weights is not None and len(scores) != len(weights):
            raise RuntimeError(
                f'The number of scores and weights should be equal, but found the fronter {len(scores)} and the latter {len(weights)}'
            )
        if weights is None:
            if self.num == 0:
                weights = []
            else:
                weights = [1 / self.num] * self.num
        self.weights = weights

    def __call__(self, images: Tensor, labels: LongTensor) -> Tensor:
        scores = torch.zeros((len(images),))
        for i in range(self.num):
            scores += self.weights[i] * self.scores[i](images, labels)
        return scores
