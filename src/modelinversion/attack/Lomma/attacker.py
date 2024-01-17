
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from modelinversion.attack.KEDMI.config import KEDMIAttackConfig

from ..base import BaseAttacker
from ..GMI.code.generator import Generator
from ..GMI.code.discri import DGWGAN
from ..GMI.code.recovery import inversion
from .config import LommaGMIAttackConfig, LommaKEDMIAttackConfig
from ...foldermanager import FolderManager
from ...models import *
from ..KEDMI.attacker import KEDMIAttacker


class LommaGMIAttacker(BaseAttacker):
    
    def __init__(self, config: LommaGMIAttackConfig) -> None:
        super().__init__(config)
        self.config: LommaGMIAttackConfig
        
    def get_tag(self) -> str:
        config: LommaGMIAttackConfig = self.config
        return f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}'
        
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
        
        
class LommaKEDMIAttacker(KEDMIAttacker):
    
    def __init__(self, config: LommaKEDMIAttackConfig) -> None:
        super().__init__(config)
        self.config: LommaKEDMIAttackConfig
        
    def get_tag(self) -> str:
        return super().get_tag()
    
    def prepare_attack(self):
        config: LommaGMIAttackConfig = self.config
        
        # print('prepare GAN')
        # self.G = Generator(100).to(config.device)
        # self.D = DGWGAN(3).to(config.device)
        
        # self.folder_manager.load_state_dict(self.G, 
        #                            ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_G.tar'],
        #                            device=config.device)
        # self.folder_manager.load_state_dict(self.D, 
        #                            ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_D.tar'],
        #                            device=config.device)
        super().prepare_attack()
        
        print('prepare augment models')
        self.aug_models = []
        for model_name in config.aug_model_names:
            model = get_model(model_name, config.dataset_name, device=config.device)
            self.folder_manager.load_state_dict(model, ['Lomma', f'{config.aug_model_dataset_name}_{config.target_name}_{model_name}.pth'], device=config.device)
            self.aug_models.append(model)
            
        print('prepare features')
        
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
        
    def get_reg_loss(self, featureT):
        
        fea_mean = self.T_preg['mean']
        fea_std = self.T_preg['std']
    
        fea_reg = self.reparameterize(fea_mean, fea_std)
        fea_reg = fea_mean.repeat(featureT.shape[0],1)
        loss_reg = torch.mean((featureT - fea_reg).pow(2))
        # print('loss_reg',loss_reg)
        return loss_reg
        
    def get_loss(self, fake, iden):
        # return super().get_iden_loss(fake, iden)
        _, D_label = self.D(fake)
        pred = self.T(fake).result
        
        Prior_Loss = torch.mean(F.softplus(self.log_sum_exp(D_label))) - torch.mean(self.log_sum_exp(D_label))
        
        loss_fn = nn.NLLLoss()
        
        iden_loss = 0
        reg_loss = 0
        T_res = self.T(fake)
        iden_loss += F.nll_loss(T_res.result, iden)
        # if T_feature_means is not None and T_feature_stds is not None:
        reg_loss += self.config.reg_coef * self.get_reg_loss(T_res.feat[-1])
        
        for aug_model in self.aug_models:
            res = aug_model(fake).result
            iden_loss += loss_fn(res, iden)
            
        iden_loss = iden_loss / (len(self.aug_models) + 1) + reg_loss
        
        Total_Loss = Prior_Loss + self.config.coef_iden_loss * iden_loss
        
        return {
            'total': Total_Loss,
            'prior': Prior_Loss,
            'iden': iden_loss
        }
        
        
        
        
    def attack_step(self, iden) -> dict:
        return super().attack_step(iden)