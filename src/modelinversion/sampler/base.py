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
    
    def __init__(self, input_size: int | Sequence[int]) -> None:
        super().__init__()
        
        if isinstance(input_size, int):
            input_size = (input_size, )
        if not isinstance(input_size, tuple):
            input_size = tuple(input_size)
        self._input_size = input_size
        
    def __call__(self, labels: list[int], sample_num: int):
       size = (sample_num, ) + self._input_size
       
       return {label: torch.randn(size) for label in labels}
   
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
            conf, _ = model(trans).softmax(dim=-1).cpu()
            scores += conf
        res = scores / total_num
    else:
        conf, _ = model(images).softmax(dim=-1).cpu()
        res = conf
        
    return res[:, labels]
   
class ImageAugmentSelectLatentsSampler(SimpleLatentsSampler):
    
    def __init__(self, input_size: int | Sequence[int], batch_size: int, all_sample_num: int, generator: BaseImageGenerator, classifier: BaseImageClassifier, device: torch.device, create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None) -> None:
        super().__init__(input_size)
        
        self.all_sample_num = all_sample_num
        self.batch_size = batch_size
        self.generator = generator
        self.classifier = classifier
        self.device = device
        self.create_aug_images_fn = create_aug_images_fn
        
    def _get_score(self, latents: Tensor, labels: list[int]):
        
        images = self.generator(latents.to(self.device))
        return cross_image_augment_scores(self.classifier, self.device, self.create_aug_images_fn, images, labels)
        
        
    def __call__(self, labels: list[int], sample_num: int):
        all_sample_num = self.all_sample_num
        if sample_num > self.all_sample_num:
            all_sample_num = sample_num
            
        raw_size = (all_sample_num, ) + self._input_size
        latents = torch.randn(raw_size)
        scores = batch_apply(self.generator, latents, batch_size=self.batch_size, labels=labels, use_tqdm=True)
        
        results = {}
        for label in labels:
            scores_label = scores[:, label]
            _, indices = torch.topk(scores_label, k=sample_num)
            results[label] = latents[indices].detach()
        return results