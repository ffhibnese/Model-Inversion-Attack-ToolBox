
import torch
from dataclasses import dataclass
import os
import time
from torchvision.utils import save_image
from torch import nn
from torch.nn import Module
from .models import *
from .utils.log import Tee

@dataclass
class DirnameConfig:
    ckpt_dir: str
    dataset_dir: str
    cache_dir: str
    result_dir: str
    defense_ckpt_dir: str
    
    
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
    },
    'FaceScrub':{
        'MobileNet':'FaceScrub-MobileNet-Train_Acc0.9736-Val_Acc0.9613.pth',
        'BackboneMobileFaceNet':'FaceScrub-BackboneMobileFaceNet-Epoch4-Train_Acc0.992-Val_Acc0.971.pth'
    }  
}

# DEFENSE_TYPES = [
#     'no_defense',
#     'vib',
#     'bido',
#     'tl'
# ]

class FolderManager:
    """
    Manage nessary folders
    
    Features:
        store the necessary folders
        save and load state dict
        save result images
        manage print and log
    """
    
    def __init__(self, attack_ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir=None, defense_type = 'no_defense', **kwargs) -> None:
        
        # if defense_type not in DEFENSE_TYPES:
        #     raise RuntimeError(
        #         f'your defense type `{defense_type}` is not valid. Valid choices are {str(DEFENSE_TYPES)}'
        #     )

        self.config = DirnameConfig(attack_ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir)
        
        for k, v in kwargs.items():
            setattr(self.config, k, v)
        for v in self.config.__dict__.values():
            if v is not None:
                os.makedirs(v, exist_ok=True)
        
        now_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        log_filename = os.path.join(result_dir, f'{defense_type}_{now_time}.log')
        print (f'log file is placed in {log_filename}')
        self.__tee = Tee(log_filename, 'w')
            
        self.temp_cnt = 0
        
        self.defense_type = defense_type
        
    def __missing__(self, key):
        return self.config.__dict__[key]
    
    def _get_ckpt_root_dir(self, defense_type=None):
        """get root folder of checkpoints

        Args:
            defense_type (str, optional): if it is None, it will be `no_defense`. Defaults to None.

        Returns:
            str: root folder of checkpoints
        """
        # if defense_type is None:
        #     defense_type = self.defense_type
        if defense_type is None or defense_type == 'no_defense':
            return self.config.ckpt_dir
        else:
            res = os.path.join(self.config.defense_ckpt_dir,  defense_type)
            os.makedirs(res, exist_ok=True)
            return res
        
    def load_state_dict(self, model: Module, relative_paths, device, defense_type=None, state_dict_key = None):
        """
            Base function to load state dict for model. 
        

        Args:
            model (Module): The model to load state dict.
            relative_paths (str | list | None): relative paths to the checkpoint. 
            device (str): device of the model.
            defense_type (str, optional): if it is None, it will follow self.defense_type. Defaults to None.
            state_dict_keys (str, optional): the key of the state dict. It  should be in ['state_dict', 'model', 'g_ema', 'map'] if `state_dict_key` is not provided.
            
        Consider if the instance of FolderManage `folder_manager`, and the checkpoint is located in <ckpt_dir>/maomao/moew/vgg16.pt.
            
        Example::
            >>> model = VGG16(1000)
            >>> relative_path = ['maomao', 'moew', 'vgg16.pt']
            >>> device = 'cuda:0'
            >>> defense_type = 'no_defense'
            >>> folder_manager.load_state_dict(model, relative_paths, device, defense_type)
            
        Consider if the checkpoint is located in <defense_ckpt_dir>/<defense_type>/maomao/moew/vgg16_bido.pt, and the defense type is 'bido'
            
        Example::
            >>> model = VGG16(1000)
            >>> relative_path = ['maomao', 'moew', 'vgg16_bido.pt']
            >>> device = 'cuda:0'
            >>> defense_type = 'bido'
            >>> folder_manager.load_state_dict(model, relative_paths, device, defense_type)
        """
        
        # prepare path
        if relative_paths is None:
            relative_paths = []
        elif isinstance(relative_paths, str):
            relative_paths = [relative_paths]
        root_dir = self._get_ckpt_root_dir(defense_type)
        path = os.path.join(root_dir, *relative_paths)
        
        if not os.path.exists(path):
            raise RuntimeError(f'path {path} is not exist')
        
        # prepare state dict
        print(f'load {model.__class__.__name__} state dict from {path}')
        state_dict = torch.load(path, map_location=device)
        if state_dict_key is not None:
            state_dict = state_dict[state_dict_key]
        elif isinstance(state_dict, dict):
            for key in ['state_dict', 'model', 'g_ema', 'map']:
                if key in state_dict.keys():
                    state_dict = state_dict[key]
                    break
        # load state dict
        model.load_state_dict(state_dict, strict=True)
        
    def load_target_model_state_dict(self, target_model, dataset_name, target_name, device, defense_type=None, state_dict_key=None):
        
        if defense_type is None:
            defense_type = 'no_defense'
            
        if defense_type == 'no_defense':
            try:
                target_filename = target_eval_models_file[dataset_name][target_name]
            except:
                target_filename = f'{target_name}_{dataset_name}.pt'
            self.load_state_dict(target_model, ['target_eval', dataset_name, target_filename], device, defense_type=defense_type, state_dict_key=state_dict_key)
        else:
            target_filename = f'{target_name}_{dataset_name}_{defense_type}.pt'
            self.load_state_dict(target_model, [dataset_name, target_filename], device, defense_type=defense_type, state_dict_key=state_dict_key)
        
    def save_state_dict(self, model: nn.Module, relative_paths, defense_type='no_defense'):
        """
            Base function to save the state dict of model. 
        

        Args:
            model (Module): The model to load state dict.
            relative_paths (str | list | None): relative paths to the checkpoint. 
            device (str): device of the model.
            defense_type (str, optional): if it is None, it will follow self.defense_type. Defaults to None.
            
        Consider if the instance of FolderManage `folder_manager`, and the checkpoint of the model will be save in <ckpt_dir>/maomao/moew/vgg16.pt.
            
        Example::
            >>> model = VGG16(1000)
            >>> relative_path = ['maomao', 'moew', 'vgg16.pt']
            >>> device = 'cuda:0'
            >>> defense_type = 'no_defense'
            >>> folder_manager.save_state_dict(model, relative_paths, defense_type)
            
        Consider if the checkpoint of the model will be save in <defense_ckpt_dir>/<defense_type>/maomao/moew/vgg16_bido.pt, and the defense type is 'bido'
            
        Example::
            >>> model = VGG16(1000)
            >>> relative_path = ['maomao', 'moew', 'vgg16_bido.pt']
            >>> device = 'cuda:0'
            >>> defense_type = 'bido'
            >>> folder_manager.save_state_dict(model, relative_paths, defense_type)
        """
        
        root_dir = self._get_ckpt_root_dir(defense_type)
        dirname = os.path.join(root_dir, *relative_paths[:-1])
        os.makedirs(dirname, exist_ok=True)
        # nn.DataParallel()
        if isinstance(model, nn.DataParallel):
            model = model.module
        torch.save({'state_dict': model.state_dict()}, os.path.join(dirname, relative_paths[-1]))
        
    def save_target_model_state_dict(self, target_model, dataset_name, target_name):
        
        if self.defense_type == 'no_defense':
            target_filename = f'{target_name}_{dataset_name}.pt'
            self.save_state_dict(target_model, ['target_eval', dataset_name, target_filename], defense_type=self.defense_type)
        else:
            target_filename = f'{target_name}_{dataset_name}_{self.defense_type}.pt'
            
            self.save_state_dict(target_model, [dataset_name, target_filename], defense_type=self.defense_type)
    
    def get_result_folder(self, folder_name='all_imgs', save_dst='result'):
        if save_dst == 'result':
            save_root_dir = self.config.result_dir
        elif save_dst == 'cache':
            save_root_dir = self.config.cache_dir
        else:
            raise RuntimeError(f'save_dst must be `result` or `cache`')
        
        return os.path.join(save_root_dir, folder_name)
    
    def save_result_image(self, img: torch.Tensor, label: int, save_name = None, folder_name='all_imgs', save_dst='result', save_tensor=False):
        """save images

        Args:
            img (torch.Tensor): image of the tensor
            label (int): target label
            save_name (str, optional): names of the save images. Defaults to None.
            folder_name (str, optional): the folder that images save to. Defaults to 'all_imgs'.
            save_dst (str, optional): the kind of save place, it should be `result` or `cache`. Defaults to 'result'.
        """
        
        if save_dst == 'result':
            save_root_dir = self.config.result_dir
        elif save_dst == 'cache':
            save_root_dir = self.config.cache_dir
        else:
            raise RuntimeError(f'save_dst must be `result` or `cache`')
            
        if isinstance(label, torch.Tensor):
            label = label.item()
        save_dir = os.path.join(save_root_dir, folder_name, f'{label}')
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f'{self.temp_cnt}.jpg'
            save_tensor_name = f'{self.temp_cnt}.pt'
            self.temp_cnt += 1
        save_path = os.path.join(save_dir, save_name)
        save_tensor_path = os.path.join(save_dir, save_tensor_name)
        if save_tensor:
            torch.save(img, save_tensor_path)
        save_image(img.detach(), save_path, normalize=True)
        return save_path
    
    def save_result_images(self, imgs: torch.Tensor, labels: list, save_names: list = None, folder_name='all_imgs', save_dst = 'result', save_tensor=False):
        """save images

        Args:
            imgs (torch.Tensor): image of the tensor
            labels (list | int): target labels
            save_names (list[str], optional): names of the save images. Defaults to None.
            folder_name (str, optional): the folder that images save to. Defaults to 'all_imgs'.
            save_dst (str, optional): the kind of save place, it should be `result` or `cache`. Defaults to 'result'.
        """
        
        if isinstance(labels, int):
            labels = [labels] * len(imgs)
        
        for i in range(len(labels)):
            save_name = None if save_names is None else save_names[i]
            self.save_result_image(imgs[i], labels[i], save_name=save_name, folder_name=folder_name, save_dst=save_dst, save_tensor=save_tensor)
            