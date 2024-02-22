
from ..base import BaseAttacker
from .code.recovery import inversion
from .code.generator import Generator
from .code.discri import DGWGAN
from .config import GMIAttackConfig
from ...foldermanager import FolderManager
from ...models import *




class GMIAttacker(BaseAttacker):
    
    def __init__(self, config: GMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}'
        super().__init__(config)
        self.config: GMIAttackConfig
        
    def get_tag(self) -> str:
        return self._tag
        
    def prepare_attack(self):
        config: GMIAttackConfig = self.config
        self.G = Generator(100).to(config.device)
        self.D = DGWGAN(3).to(config.device)
        
        self.folder_manager.load_state_dict(self.G, 
                                   ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_G.tar'],
                                   device=config.device)
        self.folder_manager.load_state_dict(self.D, 
                                   ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_D.tar'],
                                   device=config.device)
        
    def attack_step(self, iden):
        return inversion(self.config, self.G, self.D, self.T, self.E, iden, self.folder_manager)
        