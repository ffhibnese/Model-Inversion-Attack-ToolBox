
import torch
from dataclasses import dataclass
import os
from torchvision.utils import save_image
from torch import nn
from ..models import *
from .log import Tee

@dataclass
class DirnameConfig:
    ckpt_dir: str
    dataset_dir: str
    cache_dir: str
    result_dir: str
    
    
target_eval_models_file = {
    'celeba': {
        'vgg16': 'VGG16_88.26.tar',
        'ir152': 'IR152_91.16.tar',
        'facenet64': 'FaceNet64_88.50.tar',
        'facenet': 'FaceNet_95.88.tar'
    },
    'vggface2':{
        'resnet50_scratch_dag': 'resnet50_scratch_dag.pth',
        'inception_resnetv1': '20180402-114759-vggface2.pt'
    }
        
}

class FolderManager:
    
    def __init__(self, ckpt_dir, dataset_dir, cache_dir, result_dir, **kwargs) -> None:
        self.config = DirnameConfig(ckpt_dir, dataset_dir, cache_dir, result_dir)
        for k, v in kwargs.items():
            setattr(self.config, k, v)
        for v in self.config.__dict__.values():
            if v is not None:
                os.makedirs(v, exist_ok=True)
            
        self.__tee = Tee(os.path.join(result_dir, 'attack.log'), 'w')
            
        self.temp_cnt = 0
        
    def load_state_dict(self, model: nn.Module, relative_paths, device):
        
        if isinstance(relative_paths, str):
            relative_paths = [relative_paths]
        path = os.path.join(self.config.ckpt_dir, *relative_paths)
        
        if not os.path.exists(path):
            raise RuntimeError(f'path `{path}` is NOT EXISTED!')
        state_dict = torch.load(path, map_location=device)
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=True)
        
    def save_state_dict(self, model: nn.Module, relative_paths):
        dirname = os.path.join(self.config.ckpt_dir, *relative_paths[:-1])
        os.makedirs(dirname, exist_ok=True)
        # nn.DataParallel()
        if isinstance(model, nn.DataParallel):
            model = model.module
        torch.save({'state_dict': model.state_dict()}, os.path.join(dirname, relative_paths[-1]))
        
    def save_target_model_state_dict(self, target_model, dataset_name, target_name):
        target_filename = f'{target_name}_{dataset_name}.pt'
        self.save_state_dict(target_model, ['target_eval', dataset_name, target_filename])
        
    def load_target_model_state_dict(self, target_model, dataset_name, target_name, device):
        try:
            target_filename = target_eval_models_file[dataset_name][target_name]
        except:
            target_filename = f'{target_name}_{dataset_name}.pt'
        self.load_state_dict( target_model, ['target_eval', dataset_name, target_filename], device)
            
    def save_result_image(self, img: torch.Tensor, label: int, save_name = None, folder_name='all_imgs'):
        if isinstance(label, torch.Tensor):
            label = label.item()
        save_dir = os.path.join(self.config.result_dir, folder_name, f'{label}')
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f'{self.temp_cnt}.jpg'
            self.temp_cnt += 1
        save_path = os.path.join(save_dir, save_name)
        save_image(img.detach(), save_path, normalize=True)
        
    def save_result_images(self, imgs: torch.Tensor, labels: list, save_names = None, folder_name='all_imgs'):
        
        for i in range(len(labels)):
            save_name = None if save_names is None else save_names[i]
            self.save_result_image(imgs[i], labels[i], save_name=save_name, folder_name=folder_name)
            
class DefenseFolderManager(FolderManager):
    
    def __init__(self, attack_ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type = 'no_defense', **kwargs) -> None:
        super().__init__(attack_ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir = defense_ckpt_dir, **kwargs)
        
        self.defense_type = defense_type
        
    def load_defense_state_dict(self, model: nn.Module, relative_paths, device):
        
        if isinstance(relative_paths, str):
            relative_paths = [relative_paths]
        path = os.path.join(self.config.defense_ckpt_dir, *relative_paths)
        
        if not os.path.exists(path):
            raise RuntimeError(f'path `{path}` is NOT EXISTED!')
        state_dict = torch.load(path, map_location=device)
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=True)
        
    def save_target_model_state_dict(self, target_model, dataset_name, target_name, defense_type='no_defense'):
        
        if defense_type == 'no_defense':
            super().save_target_model_state_dict(target_model, dataset_name, target_name)
            return
        
        target_filename = f'{target_name}_{dataset_name}_{defense_type}.pt'
        # super().save_state_dict(target_model, [defense_type, dataset_name, target_filename])
        
        dirname = os.path.join(self.config.defense_ckpt_dir, defense_type, dataset_name)
        os.makedirs(dirname, exist_ok=True)
        # nn.DataParallel()
        if isinstance(target_model, nn.DataParallel):
            target_model = target_model.module
        torch.save({'state_dict': target_model.state_dict()}, os.path.join(dirname, target_filename))
        
    def load_target_model_state_dict(self, target_model, dataset_name, target_name, device, defense_type='no_defense'):
        
        if defense_type == 'no_defense':
            super().load_target_model_state_dict(target_model, dataset_name, target_name, device)
        else:
            target_filename = f'{target_name}_{dataset_name}_{defense_type}.pt'
            self.load_defense_state_dict(target_model, [defense_type, dataset_name, target_filename], device)
            