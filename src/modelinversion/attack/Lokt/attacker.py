import os
from dataclasses import dataclass, field

from ...foldermanager import FolderManager
from ..PLGMI.attacker import PLGMIAttackConfig, PLGMIAttacker
from ...models import *

@dataclass
class LoktAttackConfig(PLGMIAttackConfig):
    surrogate_names: list = field(default_factory=lambda: ['vgg16'])

@dataclass
class LoktAttacker(PLGMIAttacker):
    
    def __init__(self, config: LoktAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}_{config.gan_target_name}'
        super().__init__(config)
        self.config: LoktAttackConfig
        
        self.surrogate_models = []
        
    
    def prepare_attack(self):
        super().prepare_attack()
        
        config = self.config
        folder_manager = self.folder_manager
        
        self.surrogate_models = []
        
        for S_name in config.surrogate_names:
            surrogate_model = get_model(S_name, config.dataset_name, device=config.device)
            # folder_manager.load_target_model_state_dict(surrogate_model, config.dataset_name, S_name, device=config.device)
            folder_manager.load_state_dict(surrogate_model, ['lokt', f'{S_name}_{config.dataset_name}_{self.config.target_name}_{self.config.defense_type}.pt'], device=config.device)
            surrogate_model.eval()
            self.surrogate_models.append(surrogate_model)
            
    def get_loss(self, fake, iden):
        aug_list = self.config.attack_transform
        
        inv_loss = 0
        
        for S in self.surrogate_models:
            out1 = S(aug_list(fake)).result
            out2 = S(aug_list(fake)).result

            inv_loss += self.loss_fn(out1, iden) + self.loss_fn(out2, iden)
        return inv_loss / len(self.surrogate_models)
            
    def attack_step(self, iden):
        return super().attack_step(iden)