import os
import yaml
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Tuple, Callable

import torch
from torch import Tensor, LongTensor
from torchvision.utils import save_image

from ..models import *
from ..metrics import *
from ..scores import *
from ..sampler import *
from ..utils import batch_apply, print_as_yaml, print_split_line, get_random_string
from .optimize import BaseImageOptimization

@dataclass
class ImageClassifierAttackConfig:
    """This is the configuration class to store the configuration of a [`ImageClassifierAttacker`]. It is used to instantiate an Attacker according to the specified arguments. 
    
    Args:
        latents_sampler (BaseLatentsSampler): 
            The sampler for generating latent vectors for each label.
        initial_num (int, optional): 
            The number of latent vectors for each label at the time of initial selection. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        initial_latents_score_fn (BaseLatentScore, optional): 
            The function to generate scores of each latents vectors for each class. The attacker will select `optimize_num` latent vectors with the highest scores for each class in the initial selection process. No initial selection when it is `None`. Defaults to `None`.
        initial_select_batch_size (int, optional): 
            Batch size in the initial process. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        optimize_num (int): 
            The number of latent vectors for each label at the time of optimization. Defaults to 50.
        optimize_fn (BaseImageOptimization): 
            The function to generate images and final labels from initial latent vectors and labels. 
        optimize_batch_size (int): 
            Batch size in the optimization process. It will keep the same as `optimize_num` when it is `None`. Defaults to 5.
        final_num (int, optional): 
            The number of latent vectors for each label at the time of final selection. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        final_images_score_fn (BaseImageClassificationScore, optional): 
            The function to generate scores of images for the given labels. The attacker will select `final_num` images with the highest scores for the given label for each class in the final selection process. No final selection when it is `None`. Defaults to `None`.
        final_select_batch_size (int, optional): 
            Batch size in the final process. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        save_dir (str, optional):
            The file folder to save inversed images. Defaults to `None`.
        save_optimized_images (bool, optional):
            Whether to save the optimized images. Defaults to False.
        save_final_images (bool, optional):
            Whether to save the final images. Defaults to False.
        save_kwargs (dict, optional):
            Other args for saving images. Defaults to `{}`.
        eval_metrics (list[ImageMetric], optional): 
            A list of metric methods. Defaults to `[]`.
        eval_optimized_result (bool, optional):
            Whether to evaluate the optimized results. Defaults to False.
        eval_final_result (bool, optional):
            Whether to evaluate the final results. Defaults to False.
    """
    
    # sample latent
    latents_sampler: BaseLatentsSampler
    
    # # initial selection
    initial_num: Optional[int] = None
    initial_latents_score_fn: Optional[BaseLatentScore] = None
    initial_select_batch_size: Optional[int] = None
    
    # optimzation & generate images
    optimize_num: int = 50
    optimize_fn: BaseImageOptimization = None
    optimize_batch_size: int = 5
    
    # final selection
    final_num: Optional[int] = None
    final_images_score_fn: Optional[BaseImageClassificationScore] = None
    final_select_batch_size: Optional[int] = None
    
    # save
    save_dir: Optional[str] = None
    save_optimized_images: bool = False
    save_final_images: bool = False
    save_kwargs: dict = field(default_factory=lambda: {})
    
    # metric
    eval_metrics: list[ImageMetric] = field(default=lambda: [])
    eval_optimized_result: bool = False
    eval_final_result: bool = False
    

    
class ImageClassifierAttacker(ABC):
    """This is the model inversion attack class for image classification task.

    Args:
        config (ImageClassifierAttackConfig): ImageClassifierAttackConfig.
    """
    
    def __init__(self, config: ImageClassifierAttackConfig) -> None:
        self.config = self._preprocess_config(config)
        
        self.optimized_images = []
        self.optimized_labels = []
        
    
    def _preprocess_config(self, config: ImageClassifierAttackConfig):
        
        if (config.save_optimized_images or config.save_final_images) and not config.save_dir:
            raise RuntimeError('`save_dir` is not set')
        
        if config.latents_sampler is None:
            raise RuntimeError('`latents_sampler` cannot be None')
        
        if config.optimize_fn is None:
            raise RuntimeError('`optimize_fn` cannot be None')
        
        if config.optimize_num is None:
            raise RuntimeError('`optimize_num` cannot be None')
        
        if config.initial_num is None:
            config.initial_num = config.optimize_num
            
        if config.final_num is None:
            config.final_num = config.optimize_num
            
        if config.final_num > config.optimize_num:
            warnings.warn('the final number is larger than the optimize number, automatically set the latter to the fronter')
            config.optimize_num = config.final_num
            
        if config.optimize_num > config.initial_num:
            warnings.warn('the optimize number is larger than the initial number, automatically set the latter to the fronter')
            config.initial_num = config.optimize_num
            
        if config.initial_select_batch_size is None:
            config.initial_select_batch_size = config.optimize_batch_size
            
        if config.final_select_batch_size is None:
            config.final_select_batch_size = config.final_select_batch_size
            
        return config
    
    def initial_latents(self, batch_size: int, sample_num: int, select_num: int, labels: list[int], latent_score_fn: Optional[Callable] = None) -> dict[int, Tensor]:
        
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        
        # if sample_num < select_num:
        #     warnings.warn('sample_num < select_num. set sample_num = select_num')
        #     sample_num = select_num
        
        latents_dict = self.config.latents_sampler(labels, sample_num)
        
        if latent_score_fn is None or sample_num == select_num:
            if sample_num > select_num:
                warnings.warn('no score function, automatically sample `select_num` latents')
            results = latents_dict
        else:
            results = {}
            for label in tqdm(labels):
                latents = latents[label]
                labels = torch.ones((len(latents),), dtype=torch.long)
                # scores = latent_score_fn(latents, labels)
                scores = batch_apply(latent_score_fn, latents, labels, batch_size=batch_size)
                _, indices = torch.topk(scores, k=select_num)
                results[label] = latents[indices]
                
        return self.concat_tensor_labels(results)
    
    def concat_tensor_labels(self, target_dict: dict):
        tensors = []
        labels = []
        for target, latents in target_dict.items():
            tensors.append(latents)
            labels += [target] * len(latents)
        
        labels = LongTensor(labels)
        tensors = torch.cat(tensors, dim=0)
        return tensors, labels
    
    def concat_optimized_images(self):
        optimized_images = torch.cat(self.optimized_images, dim=0)
        optimized_labels = torch.cat(self.optimized_labels, dim=0)
        return optimized_images, optimized_labels
        
    def final_selection(self, batch_size: int, final_num: int, images: Tensor, labels: LongTensor, image_score_fn: Optional[Callable]=None):
        
        assert len(images) == len(labels)
        
        if final_num != len(images) and image_score_fn is None:
            warnings.warn('no score function but final num is not equal to the number of latents')
            final_num = len(images)
                 
        if final_num == len(images):
            return images, labels
        
        print('execute final selection')
        scores = batch_apply(self, image_score_fn, images, labels, batch_size=batch_size, use_tqdm=True)
        
        targets = set(labels.tolist())
        
        results = {}
        
        for target in targets:
            indices = torch.where(labels == target)
            target_images = images[indices]
            target_scores = scores[indices]
            _, topk_idx = torch.topk(target_scores, k=final_num)
            results[target] = target_images[topk_idx]
            
        return self.concat_tensor_labels(results)
    
    def update_optimized_images(self, images: Tensor, labels: LongTensor):
        assert len(images) == len(labels)
        self.optimized_images.append(images)
        self.optimized_labels.append(labels)
    
    def batch_optimize(self, init_latents: Tensor, labels: Tensor):
        images, labels = self.config.optimize_fn(init_latents, labels)
        self.update_optimized_images(images.detach().cpu(), labels.detach().cpu())
        
        if self.config.save_optimized_images:
            self.save_images(os.path.join(self.config.save_dir, 'optimized_images'), images=images, labels = labels)
    
    def _evaluation(self, images, labels, description):
        
        result = OrderedDict()
        for metric in self.config.eval_metrics:
            for k, v in metric(images, labels).items():
                result[k] = v
                
        print_split_line(description)
        print_as_yaml(result)
        print_split_line()
        
    def save_images(self, root_dir: str, images: Tensor, labels: LongTensor):
        assert len(images) == len(labels)
        
        for i in range(len(images)):
            image = images[i].detach()
            label = labels[i].item()
            save_dir = os.path.join(root_dir, f'{label}')
            os.makedirs(save_dir, exist_ok=True)
            random_str = get_random_string(length=6)
            save_path = os.path.join(save_dir, f'{label}_{random_str}.png')
            save_image(image, save_path, **self.config.save_kwargs)
            
    
    def attack(self, target_list: list[int]):
        config = self.config
        os.makedirs(config.save_dir, exist_ok=True)
        
        # print_split_line('Attack Config')
        # print(config)
        # print_split_line()
        
        # initial selection for each target
        print('execute initial selection')
        init_latents, init_labels = self.initial_latents(
            config.initial_select_batch_size, 
            config.initial_num, 
            config.optimize_num, 
            target_list, 
            config.initial_latents_score_fn
        )

        # execute optimize
        print('execute optimization')
        batch_apply(self.batch_optimize, init_latents, init_labels, batch_size=config.optimize_batch_size, description='Optimized Batch')
        
        # concat optimized images and labels
        optimized_images, optimized_labels = self.concat_optimized_images()
            
                
        if self.config.eval_optimized_result:
            print('evaluate optimized result')
            self._evaluation(optimized_images, optimized_labels, 'Optimized Image Evaluation')
        
        # final selection
        print('execute final selection')
        final_images, final_labels = self.final_selection(config.final_select_batch_size, config.final_num, optimized_images, optimized_labels, config.final_images_score_fn)
        
        if config.save_final_images:
            print('save final images')
            self.save_images(os.path.join(config.save_dir, 'final_images'), final_images, final_labels)
        
        if self.config.eval_final_result:
            print('evaluate final result')
            self._evaluation(final_images, final_labels, 'Final Image Evaluation')
        
    

  