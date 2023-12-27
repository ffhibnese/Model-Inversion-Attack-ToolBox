
from ..base import BaseAttacker
from .code.reconstruct import inversion
from .config import PLGMIAttackConfig
from ...foldermanager import FolderManager
from .code.models.generators.resnet64 import ResNetGenerator
from ...models import *



class PLGMIAttacker(BaseAttacker):
    
    def __init__(self, config: PLGMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}_{config.gan_target_name}'
        super().__init__(config)
        self.config: PLGMIAttackConfig
        
    def get_tag(self) -> str:
        return self._tag
        
    def prepare_attack_models(self):
        config: PLGMIAttackConfig = self.config
        self.G = ResNetGenerator(num_classes=NUM_CLASSES[config.dataset_name], distribution=config.gen_distribution).to(self.config.device)
        
        self.folder_manager.load_state_dict(
            self.G, 
            ['PLGMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_PLG_MI_G.tar'], 
            device=config.device
        )
        
    def attack_step(self, iden):
        acc, acc_5, acc_var, acc_var5 = inversion(
            self.config, self.G, self.T, self.E, iden, self.folder_manager, 
            self.config.lr, self.config.iter_times, 5
        )
        
        return {
            'acc': acc,
            'acc5': acc_5,
            'acc_var': acc_var,
            'acc5_var': acc_var5
        }
        