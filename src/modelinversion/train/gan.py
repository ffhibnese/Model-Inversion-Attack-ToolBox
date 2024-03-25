import os
import importlib
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Optional, Iterator, Tuple, Callable, Sequence
import math

import torch
from torch import nn, Tensor, LongTensor
from torch.nn import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from ..models import BaseImageGenerator, PlgmiGenerator64, PlgmiGenerator256, PlgmiDiscriminator64, PlgmiDiscriminator256, BaseImageClassifier, BaseLatentsSampler
from ..utils import unwrapped_parallel_module, ClassificationLoss, obj_to_yaml

def train_gan(
            max_iters: int, 
            dataloader: Iterator[Tensor | Tuple[Tensor, LongTensor]],
            train_generator_step: Callable[[int, Iterator], OrderedDict],
            train_discriminator_step: Callable[[int, Iterator], OrderedDict],
            save_fn: Callable[[int], None],
            save_ckpt_iters: int = 1000,
            show_train_info_iters: Optional[int] = 1000,
            before_iters_step: Callable[[int], None] = None,
            ncritic=5
            ):
    
    bar = tqdm(range(max_iters), leave=False)
    
    for i in bar:
        
        if before_iters_step is not None:
            before_iters_step(i)
        
        if i % ncritic == 0:
            gen_infos = train_generator_step(i, dataloader)
        
        dis_infos = train_discriminator_step(i, dataloader)
        
        if (i+1) % save_ckpt_iters == 0:
            save_fn(i)
            
        if show_train_info_iters is not None and (i+1) % show_train_info_iters == 0:
            s = obj_to_yaml(OrderedDict(iters=(i+1), generator=gen_infos, discriminator=dis_infos))
            bar.write(s)
    save_fn(max_iters)
            
class GanTrainer(ABC):
    
    def __init__(self,
                 experiment_dir: str,
                 batch_size: str,
                 generator: BaseImageGenerator, 
                 discriminator: Module,
                 device: torch.device,
                 gen_optimizer: Optimizer,
                 dis_optimizer: Optimizer,
                #  gen_optimizer_class: str | type[Optimizer],
                #  gen_optimizer_kwargs: dict,
                #  dis_optimizer_class: str | type[Optimizer],
                #  dis_optimizer_kwargs: dict,
                 save_ckpt_iters: int,
                 show_images_iters: Optional[int] = None,
                 show_train_info_iters: Optional[int] = None,
                 ncritic: int = 5
                 ) -> None:
        self.experiment_dir = experiment_dir
        self.save_img_dir = os.path.join(experiment_dir, 'gen_images')
        os.makedirs(experiment_dir, exist_ok=True)
        if show_images_iters is not None:
            os.makedirs(self.save_img_dir, exist_ok=True)
            
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.save_ckpt_iters = save_ckpt_iters
        self.show_images_iters = show_images_iters
        self.ncritic = ncritic
        self.show_train_info_iters = show_train_info_iters
        self.batch_size = batch_size
            
        # optim_module = importlib.import_module('torch.optim')
        # if isinstance(gen_optimizer_class, str):
        #     gen_optimizer_class = getattr(optim_module, gen_optimizer_class)
        # if isinstance(dis_optimizer_class, str):
        #     dis_optimizer_class = getattr(optim_module, dis_optimizer_class)
        # self.gen_optimizer = optim_module(self.generator.parameters(), **gen_optimizer_kwargs)
        # self.dis_optimizer = optim_module(self.discriminator.parameters(), **dis_optimizer_kwargs)
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
            
    def save_checkpoint(self, iters: int):
        for name, module in [
            ['G', self.generator],
            ['D', self.discriminator],
            ['G_optim', self.gen_optimizer],
            ['D_optim', self.dis_optimizer]
        ]:
                
            save_path = os.path.join(self.experiment_dir, f'{name}.pth')
            torch.save({'state_dict': unwrapped_parallel_module(module).state_dict()}, save_path)
            
    @abstractmethod
    def sample_images(self, num: int):
        pass
        
    @abstractmethod
    def train_gen_step(self, iters: int, dataloader: Iterator[Tensor | Tuple[Tensor, LongTensor]]):
        pass
    
    @abstractmethod
    def train_dis_step(self, iters: int, dataloader: Iterator[Tensor | Tuple[Tensor, LongTensor]]):
        pass
    
    def before_iter_step(self, iters: int):
        self.generator.train()
        self.discriminator.train()
        
        with torch.no_grad():
            if self.show_images_iters is not None and (iters+1)%self.show_images_iters == 0:
                save_image_num = min(self.batch_size, 16)
                images = self.sample_images(save_image_num).detach().cpu()
                nrow = int(math.sqrt(save_image_num))
                save_path = os.path.join(self.save_img_dir, f'iter_{iters+1}.png')
                save_image(images, save_path, nrow=nrow, normalize=True)
            
    def train(self, dataloader: Iterator[Tensor | Tuple[Tensor, LongTensor]], max_iters: int):
        train_gan(
            max_iters,
            dataloader,
            self.train_gen_step,
            self.train_dis_step,
            self.save_checkpoint,
            self.save_ckpt_iters,
            self.show_train_info_iters,
            self.before_iter_step,
            self.ncritic
        )
        
class PlgmiGanTrainer(GanTrainer):
    
    def __init__(self, experiment_dir: str, 
                 batch_size: str, 
                 latents_sampler: BaseLatentsSampler,
                 generator: PlgmiGenerator64 | PlgmiGenerator256, 
                 discriminator: PlgmiDiscriminator64 | PlgmiDiscriminator256,
                 num_classes: int,
                 target_model: BaseImageClassifier, 
                 classification_loss_fn: str | Callable,
                 device: torch.device, 
                 augment: Optional[Callable],
                 gen_optimizer: Optimizer,
                 dis_optimizer: Optimizer,
                 save_ckpt_iters: int, 
                 show_images_iters: int | None = None, 
                 show_train_info_iters: int | None = None, 
                 ncritic: int = 5) -> None:
        super().__init__(experiment_dir, batch_size, generator, discriminator, device, gen_optimizer, dis_optimizer, save_ckpt_iters, show_images_iters, show_train_info_iters, ncritic)
        
        self.num_classes = num_classes
        self.generator: PlgmiGenerator64 | PlgmiGenerator256
        self.discriminator: PlgmiDiscriminator64 | PlgmiDiscriminator256
        self.target_model = target_model
        self.augment = augment
        self.classification_loss = ClassificationLoss(classification_loss_fn)
        
        self.latents_sampler = latents_sampler
        
    def sample_images(self, num: int):
        latents = self.latents_sampler(num, self.batch_size).to(self.device)
        labels = torch.randint(0, self.num_classes, (len(latents), ), dtype=torch.long, device=self.device)
        fake = self.generator(latents, labels = labels)
        return fake
    
    def sample_fake(self):
        
        latents = self.latents_sampler(self.batch_size, self.batch_size).to(self.device)
        labels = torch.randint(0, self.num_classes, (len(latents), ), dtype=torch.long, device=self.device)
        fake = self.generator(latents, labels = labels)
        return fake, labels
    
    def train_gen_step(self, iters: int, dataloader: Iterator[Tensor | Tuple[Tensor | LongTensor]]):
        fake, pseudo_y = self.sample_fake()
        dis_fake = self.discriminator(fake, pseudo_y)

        aug_fake = self.augment(fake) if self.augment is not None else fake
        target_pred = self.target_model(aug_fake)
        inv_loss = self.classification_loss(target_pred, pseudo_y).mean()
        
        dis_loss = - torch.mean(dis_fake)
        
        gen_loss = dis_loss + 0.2 * inv_loss
        
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return OrderedDict([
            ['dis_loss', dis_loss.item()],
            ['inv_loss', inv_loss.item()],
            ['loss', gen_loss.item()]
        ])
        
    def train_dis_step(self, iters: int, dataloader: Iterator[Tensor | Tuple[Tensor | LongTensor]]):
        fake, pseudo_y = self.sample_fake()
        dis_fake = self.discriminator(fake, pseudo_y)
        
        realds = next(dataloader)
        if not isinstance(realds, Sequence) or len(realds) != 2:
            raise RuntimeError(f'the item of the dataloader are expected to (images, labels)')
        
        real, y = realds[0].to(self.device), realds[1].to(self.device)
        
        dis_real = self.discriminator(real, y)
        
        dis_fake_loss = torch.mean(torch.relu(1 + dis_fake))
        dis_real_loss = torch.mean(torch.relu(1 - dis_real))
        
        loss = dis_fake_loss + dis_real_loss
        
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        
        return OrderedDict([
            ['real_loss', dis_real_loss.item()],
            ['fake_loss', dis_fake_loss.item()],
            ['loss', loss.item()]
        ])