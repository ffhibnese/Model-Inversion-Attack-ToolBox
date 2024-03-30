import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable, Sequence

import torch
from torch import Tensor, LongTensor

from ..models import BaseImageGenerator, BaseImageClassifier
from ..utils import batch_apply

class BaseLatentsSampler(ABC):
    """The base class for latent vectors samplers.
    """
    
    @abstractmethod
    def __call__(self, labels: list[int], sample_num: int):
        """The sampling function of the sampler.

        Args:
            sample_num (int): The number of latent vectors sampled.
            batch_size (int): Batch size for sampling.
        """
        pass
    
class SimpleLatentsSampler(BaseLatentsSampler):
    """A latent vector sampler that generates Gaussian distributed random latent vectors with the given `input_size`.

    Args:
        input_size (int or Sequence[int]): The shape of the latent vectors.
    """
    
    def __init__(self, input_size: int | Sequence[int], batch_size: int, latents_mapping: Optional[Callable] = None) -> None:
        super().__init__()
        
        if isinstance(input_size, int):
            input_size = (input_size, )
        if not isinstance(input_size, tuple):
            input_size = tuple(input_size)
        self._input_size = input_size
        self.batch_size = batch_size
        self.latents_mapping = latents_mapping
        
    def get_batch_latent_size(self, batch_num: int):
        return (batch_num, ) + self._input_size
        
    def __call__(self, labels: list[int], sample_num: int):
       size = self.get_batch_latent_size(sample_num)
       
       latents = torch.randn(size)
       if self.latents_mapping is not None:
           latents = self.latents_mapping(latents)
       return {label: latents.detach().clone() for label in labels}

   
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
   
class ImageAugmentSelectLatentsSampler(SimpleLatentsSampler):
    
    def __init__(self, input_size: int | Sequence[int], batch_size: int, all_sample_num: int, generator: BaseImageGenerator, classifier: BaseImageClassifier, device: torch.device, latents_mapping: Optional[Callable] = None, create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None) -> None:
        super().__init__(input_size, batch_size, latents_mapping)
        
        self.all_sample_num = all_sample_num
        self.generator = generator
        self.classifier = classifier
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn
        
    def _get_score(self, latents: Tensor, labels: list[int]):
        
        images = self.generator(latents.to(self.device))
        return cross_image_augment_scores(self.classifier, self.device, self.create_aug_images_fn, images)
        
        
    def __call__(self, labels: list[int], sample_num: int):
        all_sample_num = self.all_sample_num
        if sample_num > self.all_sample_num:
            all_sample_num = sample_num
            
        raw_size = self.get_batch_latent_size(all_sample_num)
        latents = torch.randn(raw_size)
        if self.latents_mapping is not None:
            latents = batch_apply(self.latents_mapping, latents, batch_size=self.batch_size)
        scores = batch_apply(self._get_score, latents, batch_size=self.batch_size, labels=labels, use_tqdm=True)
        
        results = {}
        for label in labels:
            scores_label = scores[:, label]
            _, indices = torch.topk(scores_label, k=sample_num)
            results[label] = latents[indices].detach()
        return results