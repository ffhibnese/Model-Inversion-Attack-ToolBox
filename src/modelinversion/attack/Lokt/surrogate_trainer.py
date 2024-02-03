import os
from typing import Callable
from dataclasses import field, dataclass

import kornia
import torch
from torch import nn
from torch._C import LongTensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modelinversion.models import ModelResult

from ...foldermanager import FolderManager
from ...models import get_model, BaseTargetModel, NUM_CLASSES
from ...trainer import BaseTrainArgs, BaseTrainer
from ..PLGMI.code.m_cgan import ResNetGenerator
from .models.discri import SNResNetConditionalDiscriminator
from torch.utils.data import DataLoader, TensorDataset

class LoktSurrogateTrainArgs(BaseTrainArgs):
    augment: Callable = field(default_factory=
        kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        )
    )
    
    sample_batch_size: int = 100
    samples_per_class: int = 500
    target_name: str = 'vgg16'
    target_dataset_name: str = 'celeba'
    target_defense_type: str = 'no_defense'

class LoktSurrogateTrainer(BaseTrainer):
    
    def __init__(self, args: LoktSurrogateTrainArgs, folder_manager: FolderManager, model: BaseTargetModel, gen_z_dim, generator, discriminator, optimizer: Optimizer, lr_scheduler: LRScheduler = None, **kwargs) -> None:
        super().__init__(args, folder_manager, model, optimizer, lr_scheduler, **kwargs)
        self.args: LoktSurrogateTrainArgs
        
        self.G = generator
        self.D = discriminator
        self.gen_z_dim = gen_z_dim
        
        # create dataset
        self.num_classes = NUM_CLASSES[args.target_dataset_name]
        dataset_dir = os.path.join(self.folder_manager.config.cache_dir, 'surrogate_dataset', args.target_dataset_name)
        
        self.dataset_path = os.path.join(dataset_dir, f'{args.target_name}_{args.target_defense_type}.pt')
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(dataset_dir, exist_ok=True)
            T = get_model(args.target_name, args.target_dataset_name, args.device, defense_type=args.target_defense_type)
            self.folder_manager.load_target_model_state_dict(T, self.args.target_dataset_name, args.target_name, args.device, args.defense_type)
            
            pseudo_ys = torch.arange(0, self.num_classes, dtype=torch.long).repeat_interleave(args.samples_per_class)
            zs, pred_labels = [], []
            for start_idx in range(0, len(pseudo_ys), args.sample_batch_size):
                end_idx = min(start_idx+args.sample_batch_size, len(pseudo_ys))
                z, pred_label = self._create_batch_dataset(T, pseudo_ys[start_idx: end_idx])
                zs.append(z)
                pred_labels.append(pred_label)
            zs = torch.cat(zs, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)
            torch.save({
                'pseudo_y': pseudo_ys,
                'z': zs,
                'target_y': pred_labels
            }, self.dataset_path)
            
            del T
        
        data = torch.load(self.dataset_path, map_location='cpu')
        self.dataset = TensorDataset(data['z'], data['pseudo_y'], data['target_y'])
            
    def _sample_label(self, batch_size, labels):
        args = self.args
        z = torch.randn((len(labels), self.gen_z_dim), device=args.device)
        fake = self.G(z, labels)
        return z, fake
            
    @torch.no_grad()
    def _create_batch_dataset(self, T, labels):
        z, images = self._sample_label(labels)
        pred_probs = T(images).result
        pred_labels = torch.argmax(pred_probs, dim=-1)
        return z.cpu(), pred_labels.cpu()
    
    def get_trainset(self):
        return self.dataset
    
    def prepare_input_label(self, batch):
        
        with torch.no_grad():
            batch = [elem.to(self.args.device) for elem in batch]
            z, pseudo_y, labels = batch
            inputs = self.G(z, pseudo_y)
            if self.args.augment is not None:
                inputs_aug = self.args.augment(inputs)
                inputs = torch.cat([inputs, inputs_aug])
                labels = torch.cat([labels, labels])
                
            return inputs, labels
        
    def calc_loss(self, inputs, result: ModelResult, labels: LongTensor):
        return F.cross_entropy(result.result, labels)