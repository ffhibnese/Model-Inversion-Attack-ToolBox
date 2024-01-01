
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from ..base import BaseAttacker
from ..GMI.code.generator import Generator
from ..GMI.code.discri import DGWGAN
from ..GMI.code.recovery import inversion
from .config import LommaGMIAttackConfig
from ...foldermanager import FolderManager
from ...models import *




class LommaGMIAttacker(BaseAttacker):
    
    def __init__(self, config: LommaGMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}'
        super().__init__(config)
        self.config: LommaGMIAttackConfig
        
    def get_tag(self) -> str:
        return self._tag
        
    def prepare_attack(self):
        config: LommaGMIAttackConfig = self.config
        
        print('prepare GAN')
        self.G = Generator(100).to(config.device)
        self.D = DGWGAN(3).to(config.device)
        
        self.folder_manager.load_state_dict(self.G, 
                                   ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_G.tar'],
                                   device=config.device)
        self.folder_manager.load_state_dict(self.D, 
                                   ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_D.tar'],
                                   device=config.device)
        
        print('prepare augment models')
        self.aug_models = []
        for model_name in config.aug_model_names:
            model = get_model(model_name, config.dataset_name, device=config.device)
            self.folder_manager.load_state_dict(model, ['Lomma', f'{config.aug_model_dataset_name}_{config.target_name}_{model_name}.pth'], device=config.device)
            self.aug_models.append(model)
            
        print('prepare features')
        # self.aug_preg = []
        # for model, name in zip(self.aug_models, config.aug_model_names):
        #     save_path = os.path.join(self.folder_manager.config.cache_dir, f'{name}.pt')
        #     if not os.path.exists(save_path):
        #         self._generate_preg(model, save_path)
        #     self.aug_preg.append(torch.load(save_path, map_location=config.device))
        
        save_path = os.path.join(self.folder_manager.config.cache_dir, f'{config.target_name}.pt')
        if not os.path.exists(save_path):
            self._generate_preg(self.T, save_path)
        self.T_preg = torch.load(save_path, map_location=config.device)
            
    def _generate_preg(self, model, save_path, sample_num = 5000):
        config: LommaGMIAttackConfig = self.config
        data_path = os.path.join(self.folder_manager.config.dataset_dir, self.config.dataset_name, 'split', 'public')
        dataset = ImageFolder(data_path, transform=ToTensor())
        dataloader = DataLoader(dataset, config.preg_generate_batch_size, shuffle=True)
        
        features = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                if sample_num <= 0:
                    break
                if sample_num < len(imgs):
                    imgs = imgs[sample_num:]
                sample_num -= len(imgs)
                
                imgs = imgs.to(config.device)
                add_feature = model(imgs).feat[-1]
                features.append(add_feature.cpu())
            features = torch.cat(features, dim=0)
            
        features_mean = torch.mean(features, dim=0)
        features_std = torch.std(features, dim=0)
        
        torch.save({'mean': features_mean, 'std': features_std}, save_path)
        
        
    def attack_step(self, iden):
        return inversion(self.config, self.G, self.D, self.T, self.E, iden, self.folder_manager, self.aug_models, True, self.T_preg['mean'], self.T_preg['std'], 1.)
        