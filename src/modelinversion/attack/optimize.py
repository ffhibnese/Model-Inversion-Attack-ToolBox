
import os
import importlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Iterable

import torch
from torch import Tensor, LongTensor
from torch.optim import Optimizer, Adam
from tqdm import tqdm

from ..utils import ClassificationLoss, BaseConstraint, DictAccumulator
from ..models import BaseImageClassifier, BaseImageGenerator
from ..scores import BaseLatentScore

def get_info_description(it: int, info: OrderedDict):
    ls = [f'{k}: {v}' for k, v in info.items()]
    right_str = '  '.join(ls)
    description = f'iter {it}: {right_str}'
    return description

@dataclass
class BaseImageOptimizationConfig:
    """Base class for all optimization config classes. Handles a few parameters of the experiment settings.
    
    Args:
        experiment_dir (str): The file folder that store the intermediate and final results of the optimization. 
        device (device): The device that the optimization process run on.
    """
    experiment_dir: str
    device: torch.device

class BaseImageOptimization(ABC):
    """Base class for all optimization class. Optimize the initial latent vectors and generate the optimized images.

    Args:
        config (BaseImageOptimizationConfig): Config of the optimization.
    """
    
    def __init__(self, config: BaseImageOptimizationConfig) -> None:
        super().__init__()
        self._config = config
        os.makedirs(config.experiment_dir, exist_ok=True)
    
    @property
    def config(self):
        return self._config
    
    @abstractmethod
    def __call__(self, latents: Tensor, labels: LongTensor) -> Tuple[Tensor, LongTensor]:
        """Optimize the initial latent vectors and generate the optimized images.

        Args:
            latents (Tensor): The latent vectors of the generator.
            labels (LongTensor): The labels for the latent vectors. It has the same length with `latents`.

        Returns:
            Tuple[Tensor, LongTensor]: Returns (images, labels) that has the same length.
        """
        pass
    

@dataclass
class SimpleWhiteBoxOptimizationConfig(BaseImageOptimizationConfig):
    """Base class for all white-box optimization config classes. Handles a few parameters of gradient updating.

    Args:
        optimizer (str | type): The optimizer class.
        optimizer_kwargs (dict): Args to build the optimizer. Default to `{}`.
        iter_times (int): Update times. Defaults to 600.
        show_loss_info_iters (int): Iteration interval for displaying loss information. Default to 100.
    """
    
    optimizer: str | type = 'Adam'
    optimizer_kwargs: dict = field(default_factory=lambda: {})
    iter_times: int = 600
    show_loss_info_iters: int = 100
    
    latent_constraint: Optional[BaseConstraint] = None
    

class SimpleWhiteBoxOptimization(BaseImageOptimization):
    """Base class for all white-box optimization classes.

    Args:
        config (SimpleWhiteBoxOptimizationConfig): 
            Config of the white box optimization.
        generator (BaseImageGenerator): 
            Generator to generate images from latent vectors.
        image_loss_fn (Callable[[Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]]):
            A function to calculate loss of the generated images with given labels. Returns the loss and an optional OrderedDict that contains loss information to show.
    """
    
    def __init__(self, 
                 config: SimpleWhiteBoxOptimizationConfig, 
                 generator: BaseImageGenerator,
                 image_loss_fn: Callable[[Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]]
                 ) -> None:
        super().__init__(config)
        
        optimizer_class = config.optimizer
        if isinstance(optimizer_class, str):
            optim_module = importlib.import_module('torch.optim')
            optimizer_class = getattr(optim_module, optimizer_class, None)
            
        if not hasattr(optimizer_class, 'zero_grad'):
            raise RuntimeError('Optimizer do not has attribute `zero_grad`')
        
        if not hasattr(optimizer_class, 'step'):
            raise RuntimeError('Optimizer do not has attribute `step`')
        
        self.optimizer_class = optimizer_class
        self.generator = generator
        self.image_loss_fn = image_loss_fn
        
    def __call__(self, latents: Tensor, labels: LongTensor) -> Tuple[Tensor, LongTensor]:
        
        config: SimpleWhiteBoxOptimizationConfig = self.config
        
        latents = latents.to(config.device)
        labels = labels.to(config.device)
        latents.requires_grad_(True)
        optimizer: Optimizer = self.optimizer_class([latents], **config.optimizer_kwargs)
        
        if config.latent_constraint is not None:
            config.latent_constraint.register_center(latents)
        
        bar = tqdm(range(1, config.iter_times + 1), leave=False)
        for i in bar:
            
            fake = self.generator(latents, labels=labels)
            
            description = None
                
            loss = self.image_loss_fn(fake, labels)
            if isinstance(loss, tuple):
                loss, metric_dict = loss
                if metric_dict is not None and len(metric_dict) > 0:
                    if i == 1 or i % config.show_loss_info_iters == 0:
                        # ls = [f'{k}: {v}' for k, v in metric_dict.items()]
                        # right_str = '  '.join(ls)
                        # description = f'iter {i}: {right_str}'
                        description = get_info_description(i, metric_dict)
                        bar.set_description_str(description)
                    if i == config.iter_times:
                        # ls = [f'{k}: {v}' for k, v in metric_dict.items()]
                        description = get_info_description(i, metric_dict)
                        # right_str = '  '.join(ls)
                        # description = f'iter {i}: {right_str}'
                        # print(description)
                
                    
            if description is not None:
                print(description)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if config.latent_constraint is not None:
                latents = config.latent_constraint(latents)
            
        final_fake = self.generator(latents, labels=labels).cpu()
                
        final_labels = labels.cpu()
        
        return final_fake.detach(), final_labels.detach()
    
@dataclass
class VarienceWhiteboxOptimizationConfig(SimpleWhiteBoxOptimizationConfig):
    generate_num: int = 50
    
class VarienceWhiteboxOptimization(SimpleWhiteBoxOptimization):
    
    def __init__(self, config: VarienceWhiteboxOptimizationConfig, generator: BaseImageGenerator, image_loss_fn: Callable[[Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]]) -> None:
        super().__init__(config, generator, image_loss_fn)
        
    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
        
    def __call__(self, latents: Tensor, labels: LongTensor) -> Tuple[Tensor | LongTensor]:
        config: VarienceWhiteboxOptimizationConfig = self.config
        
        mu = latents.to(config.device)
        labels = labels.to(config.device)
        mu.requires_grad_(True)
        
        logvar = torch.ones_like(mu, requires_grad=True)
        
        optimizer: Optimizer = self.optimizer_class([mu, logvar], **config.optimizer_kwargs)
        
        bar = tqdm(range(1, config.iter_times + 1), leave=False)
        for i in bar:
            
            z = self._reparameterize(mu, logvar)
            
            fake = self.generator(z, labels=labels)
                
            loss = self.image_loss_fn(fake, labels)
            if isinstance(loss, tuple):
                loss, metric_dict = loss
                if metric_dict is not None and len(metric_dict) > 0 and (i == 1 or i % config.show_loss_info_iters == 0):
                    ls = [f'{k}: {v}' for k, v in metric_dict.items()]
                    right_str = '  '.join(ls)
                    description = f'iter {i}: {right_str}'
                    bar.set_description_str(description)
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        final_fake = []
        final_labels = []
        
        with torch.no_grad():
            for _ in range(config.generate_num):
                z = self._reparameterize(mu, logvar)
                fake = self.generator(z, labels=labels).detach().cpu()
                final_fake.append(fake)
                final_labels.append(labels.detach().cpu())
            final_fake = torch.cat(final_fake, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
            
        
        return final_fake, final_labels
    
@dataclass
class ImageAugmentWhiteBoxOptimizationConfig(SimpleWhiteBoxOptimizationConfig):
    
    loss_fn: str | Callable[[Tensor, LongTensor], Tensor] = 'cross_entropy'
    # initial_transform: Optional[Callable] = None, 
    create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]]=None
        
class ImageAugmentWhiteBoxOptimization(SimpleWhiteBoxOptimization):
    
    def __init__(self, 
                 config: ImageAugmentWhiteBoxOptimizationConfig, 
                 generator: BaseImageGenerator,
                 target_model: BaseImageClassifier
                 ) -> None:
        
        if config.create_aug_images_fn is None:
            config.create_aug_images_fn = lambda x: [x]
            
        # print(config.loss_fn)
        
        loss_fn = ClassificationLoss(config.loss_fn)

        def _image_loss_fn(images: Tensor, labels: LongTensor):
            
            # if config.initial_transform is not None:
            #     images = config.initial_transform(images)
            
            acc = 0
            loss = 0
            total_num = 0
            for aug_images in config.create_aug_images_fn(images):
                total_num += 1
                conf, _ = target_model(aug_images)
                pred_labels = torch.argmax(conf, dim=-1)
                loss += loss_fn(conf, labels)
                # print(pred_labels)
                # print(labels)
                # exit()
                acc += (pred_labels == labels).float().mean().item()
            
            return loss, OrderedDict([
                ['loss', loss.item()],
                ['target acc', acc / total_num]
            ])
            
        
        super().__init__(config, generator, _image_loss_fn)
        
@dataclass
class BrepOptimizationConfig(BaseImageOptimizationConfig):
    
    iter_times: int = 600
    
    init_sphere_radius: float = 2
    coef_sphere_expand: float = 1.3
    
    sphere_points_count: int = 32
    
    sphere_expand_score_threshold: int = 0.5
    
    step_rate: float = 0.333
    max_step_size: float = 3
    
    show_loss_info_iters: int = 100
        
class BrepOptimization(BaseImageOptimization):
    
    def __init__(self, config: BrepOptimizationConfig,
                 generator: BaseImageGenerator,
                 image_score_fn: Callable[[Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]]
                 ) -> None:
        super().__init__(config)
        
        self.generator = generator
        self.image_score_fn = image_score_fn
        
    def _generate_points_on_sphere(self, latents: Tensor, num: int, radius: Tensor, eps = 1e-5):
        # latents: (bs, zdim)
        # radius: (bs, )
        # return: (num, bs, zdim)
        
        points_shape = (num, ) + latents.shape
        vectors = torch.randn(points_shape, dtype=latents.dtype, device = latents.dtype)
        
        vectors = vectors / ((vectors ** 2).sum(dim=-1, keepdim=True).sqrt() + eps)
        return latents + vectors * radius.reshape(1, -1, 1), vectors
        
    @torch.no_grad()
    def __call__(self, latents: Tensor, labels: LongTensor) -> Tuple[Tensor, LongTensor]:
        
        config: BrepOptimizationConfig = self.config
        device = config.device
        
        bs = len(labels)
        # (bs, zdim)
        latents = latents.to(device)
        labels = labels.to(device)
        
        # (bs, )
        current_radius = torch.ones((bs, ), dtype=latents.dtype, device=device) * config.init_sphere_radius
        
        description = None
        
        for it in tqdm(1, 1+ config.iter_times, leave=False):
        
            step_size = torch.min(current_radius * config.step_rate, config.max_step_size)
            
            # (cnt, bs, zdim)
            sphere_points, sphere_point_directions = self._generate_points_on_sphere(latents, config.sphere_points_count, current_radius)
            
            accumulator = DictAccumulator()
            
            sphere_points_scores = []
            for i in range(config.sphere_points_count):
                # sphere_points_scores.append(config.latent_score_fn(sphere_points[i], labels=labels))
                batch_images = self.generator(sphere_points[i], labels=labels)
                scores = self.image_loss_fn(batch_images, labels=labels)
                if isinstance(scores, tuple):
                    scores, infos = scores
                    accumulator.add(infos)
                sphere_points_scores.append(scores)
        
            if len(accumulator) > 0 :
                if it == 1 or it % config.show_loss_info_iters == 0:
                    description = get_info_description(i, accumulator.avg())
                    print(description)
                elif it == config.show_loss_info_iters:
                    description = get_info_description(i, accumulator.avg())
                
            # (cnt, bs)
            sphere_points_scores = torch.stack(sphere_points_scores, dim=0)
            
            # (bs, zdim)
            grad_direction = (sphere_points_scores * sphere_point_directions).mean(dim = 0)
            
            latents = latents + grad_direction * step_size.reshape(-1, 1)
            
            current_radius = torch.where(
                sphere_points_scores.mean(dim=0) > config.sphere_expand_score_threshold, 
                current_radius * config.coef_sphere_expand,
                current_radius    
            )
            
        if description is not None:
            print(description)
            
        return self.generator(latents, labels=labels).detach().cpu(), labels.cpu()