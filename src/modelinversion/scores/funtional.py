import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable

import torch
from torch import Tensor, LongTensor

from ..models import BaseImageClassifier


@torch.no_grad()
def specific_image_augment_scores(model: BaseImageClassifier, device: torch.device, 
                 create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
                 images: Tensor, labels: LongTensor):
    images = images.detach().to(device)
    labels = labels.cpu()
        
    if create_aug_images_fn is not None:
        scores = torch.zeros_like(labels, dtype=images.dtype, device='cpu')
        total_num = 0
        for trans in create_aug_images_fn(images):
            total_num += 1
            conf = model(trans)[0].softmax(dim=-1).cpu()
            scores += torch.gather(conf, 1, labels.unsqueeze(1)).squeeze(1)
        return scores / total_num
    else:
        conf = model(images)[0].softmax(dim=-1).cpu()
        return torch.gather(conf, 1, labels.unsqueeze(1)).squeeze(1)
    
    
@torch.no_grad()
def cross_image_augment_scores(model: BaseImageClassifier, device: torch.device, 
                 create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
                 images: Tensor):
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