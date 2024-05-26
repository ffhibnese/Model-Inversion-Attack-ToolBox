from ..attacker import *

class VmiAttacker(ImageClassifierAttacker):
    
    def __init__(self, config: ImageClassifierAttackConfig) -> None:
        super().__init__(config)
        
    def attack(self, target_list: list[int]):
        return super().attack(target_list)