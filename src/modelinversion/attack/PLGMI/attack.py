from .reconstruct import plgmi_attack
from .config import PlgmiAttackConfig

def attack(args: PlgmiAttackConfig):
    plgmi_attack(args)