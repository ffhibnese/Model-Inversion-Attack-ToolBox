from .config import DeepInversionConfig
from .imagenet_inversion import deepinversion_attack

def attack(config: DeepInversionConfig):
    deepinversion_attack(config)