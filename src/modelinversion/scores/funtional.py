import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable

import torch
from torch import Tensor, LongTensor

from ..models import BaseImageClassifier
from ..utils import TorchLoss


# @torch.no_grad()
def specific_image_augment_scores(
    model: BaseImageClassifier,
    device: torch.device,
    create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
    images: Tensor,
    labels: LongTensor,
):
    images = images.detach().to(device)
    labels = labels.cpu()

    if create_aug_images_fn is None:
        create_aug_images_fn = lambda x: [x]

    scores = torch.zeros_like(labels, dtype=images.dtype, device='cpu')
    total_num = 0
    for trans in create_aug_images_fn(images):
        total_num += 1
        conf = model(trans)[0].softmax(dim=-1).detach().cpu()
        scores += torch.gather(conf, 1, labels.unsqueeze(1)).squeeze(1)
    return scores / total_num


def specific_image_augment_loss_score(
    model: BaseImageClassifier,
    device: torch.device,
    create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
    images: Tensor,
    labels: LongTensor,
    loss_fn: Callable,
):
    images = images.detach().to(device)
    labels = labels.to(device)

    if create_aug_images_fn is None:
        create_aug_images_fn = lambda x: [x]

    losses = torch.zeros_like(labels, dtype=images.dtype, device='cpu')
    total_num = 0
    for trans in create_aug_images_fn(images):
        total_num += 1
        losses += -loss_fn(model(trans)[0], labels).detach().cpu()
    return losses / total_num


def specific_image_augment_scores_label_only(
    model: BaseImageClassifier,
    device: torch.device,
    create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
    images: Tensor,
    labels: LongTensor,
    correct_score: float = 1,
    wrong_score=-1,
):
    images = images.detach().to(device)
    labels = labels.cpu()

    if create_aug_images_fn is not None:
        scores = torch.zeros_like(labels, dtype=images.dtype, device='cpu')
        total_num = 0
        for trans in create_aug_images_fn(images):
            total_num += 1
            correct = model(trans)[0].argmax(dim=-1).detach().cpu() == labels
            scores += torch.where(correct, correct_score, wrong_score)
        return scores / total_num
    else:
        correct = model(images)[0].argmax(dim=-1).detach().cpu() == labels
        return torch.where(correct, correct_score, wrong_score).to(images.dtype)


# @torch.no_grad()
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
