import os

import torchvision.transforms.functional as tv_f

from ..base import BaseAttacker
from .config import MirrorBlackboxAttackConfig, MirrorWhiteboxAttackConfig
from .code.presample import presample
from .code.genforce.get_genforce import get_genforce
from .code.blackbox.blackbox_attack import mirror_blackbox_attack
from .code.whitebox.whitebox_attack import mirror_white_box_attack

class MirrorBlackboxAttacker(BaseAttacker):
    
    def __init__(self, config: MirrorBlackboxAttackConfig) -> None:
        self._tag = f'blackbox_{config.dataset_name}_{config.target_name}_{config.genforce_name}'
        super().__init__(config)
        self.config: MirrorBlackboxAttackConfig
        
        presample_dir = os.path.join(self.folder_manager.config.cache_dir, 'presample')
        self.presample_dir = presample_dir
        self.register_dirs({'presample_dir': presample_dir})
        
    def get_tag(self) -> str:
        return self._tag
    
    def prepare_attack(self):
        # return super().prepare_attack()
        self.genforce_model, _ = get_genforce(self.config.genforce_name, self.config.device, self.folder_manager.config.ckpt_dir, use_discri=False)
        
        
        presample_dir = self.presample_dir
        check_presample_dir = os.path.join(presample_dir, 'img')
        if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
            print('presample')
            presample(presample_dir, self.genforce_model, batch_size=self.config.presample_batch_size, device=self.config.device)
            
    
    def attack_step(self, iden) -> dict:
        # return super().attack_step(iden)
        return  mirror_blackbox_attack(self.config, iden, self.genforce_model, self.T, self.E, self.folder_manager)
    
    
class MirrorWhiteboxAttacker(BaseAttacker):
    
    def __init__(self, config: MirrorWhiteboxAttackConfig) -> None:
        self._tag = f'whitebox_{config.dataset_name}_{config.target_name}_{config.genforce_name}'
        super().__init__(config)
        self.config: MirrorWhiteboxAttackConfig
        
        presample_dir = os.path.join(self.folder_manager.config.cache_dir, 'presample')
        self.presample_dir = presample_dir
        self.register_dirs({'presample_dir': presample_dir})
        
        to_target_transforms = None
    
        if config.dataset_name == 'celeba':
            re_size = 64
            crop = lambda x: x[..., 20:108, 20:108]

            def trans(img):
                img = tv_f.resize(img, (128,128))
                img = crop(img)
                img = tv_f.resize(img, (re_size, re_size))
                return img
        
            to_target_transforms = trans
        self.to_target_transforms = to_target_transforms
        
    def get_tag(self) -> str:
        return self._tag
    
    def prepare_attack(self):
        # return super().prepare_attack()
        self.genforce_model, _ = get_genforce(self.config.genforce_name, self.config.device, self.folder_manager.config.ckpt_dir, use_discri=False)
        
        
        presample_dir = self.presample_dir
        check_presample_dir = os.path.join(presample_dir, 'img')
        if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
            print('presample')
            presample(presample_dir, self.genforce_model, batch_size=self.config.presample_batch_size, device=self.config.device)
            
    
    def attack_step(self, iden) -> dict:
        # return super().attack_step(iden)
        # return  mirror_blackbox_attack(self.config, iden, self.genforce_model, self.T, self.E, self.folder_manager)
        return mirror_white_box_attack(self.config, self.genforce_model, self.T, self.E, self.folder_manager, iden, to_target_transforms=self.to_target_transforms)
    