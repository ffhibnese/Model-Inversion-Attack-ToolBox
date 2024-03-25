from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor, LongTensor

from .imgscore import *
from ..models import BaseImageClassifier, BaseImageGenerator

@torch.no_grad()
def cross_image_augment_scores(model: BaseImageClassifier, device: torch.device, 
                 create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]],
                 images: Tensor, labels: LongTensor | list[int]):
    images = images.detach().to(device)
    
    labels = torch.LongTensor(labels).cpu()
        
    if create_aug_images_fn is not None:
        scores = 0
        total_num = 0
        for trans in create_aug_images_fn(trans):
            total_num += 1
            conf = model(trans).softmax(dim=-1).cpu()
            scores += conf
        res = scores / total_num
    else:
        conf = model(images).softmax(dim=-1).cpu()
        res = conf
        
    return res[:, labels]

class BaseLatentScore(ABC):
    """This is a class for generating score all combinations of latents and labels.
    """
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, latents: Tensor, labels: LongTensor | list[int]) -> Tensor:
        """The scoring function to score all combinations of latents and labels.

        Args:
            latents (Tensor): Latent vectors with size (M, ...)
            labels (LongTensor | list[int]): Labels with size (N, )

        Returns:
            Tensor: The score matrix with the size of (M, N).
        """
        pass
    
class LatentClassificationAugmentConfidence(BaseLatentScore):
    """This is a class for generating score all combinations of latents and labels. The score is calculated by the conficence of the classifier model.

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
    
    def __init__(self, generator: BaseImageGenerator, model: BaseImageClassifier, device: torch.device,   
                 create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]]=None) -> None:
        self.generator = generator
        self.model = model
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn
        self.device = device
        
    @torch.no_grad()
    def __call__(self, latents: Tensor, labels: LongTensor | list[int]) -> Tensor:
        latents = latents.to(self.device)
        images = self.generator(latents, labels=labels)
        return cross_image_augment_scores(self.model, self.device, self.create_aug_images_fn, images, labels)
    
    
    
    
    
    
    
    
    
    
    