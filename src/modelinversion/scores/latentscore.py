from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor, LongTensor

from .imgscore import *
from ..models import BaseImageClassifier, BaseImageGenerator
from .funtional import specific_image_augment_scores


class BaseLatentScore(ABC):
    """This is a class for generating scores for each latent vector with the corresponding label.
    """
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, latents: Tensor, labels: LongTensor | list[int]) -> Tensor:
        """The scoring function to score all latent vectors with the corresponding labels.

        Args:
            latents (Tensor): Latent vectors.
            labels (LongTensor): The corresponding labels for latent vectors. The length of `labels` should keep the same as `images`

        Returns:
            Tensor: The score of each latent vectors.
        """
        pass
    
class LatentClassificationAugmentConfidence(BaseLatentScore):
    """This is a class for generating scores for each latent vector with the corresponding label.. The score is calculated by the conficence of the classifier model.

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
        labels = torch.LongTensor(labels).to(self.device)
        images = self.generator(latents, labels=labels)
        return specific_image_augment_scores(self.model, self.device, self.create_aug_images_fn, images, labels)
    
    
    
    
    
    
    
    
    
    
    