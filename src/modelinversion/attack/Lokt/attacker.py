import os
from dataclasses import dataclass

from ...foldermanager import FolderManager
from ..PLGMI.attacker import PLGMIAttackConfig, PLGMIAttacker

@dataclass
class LoktAttackConfig(PLGMIAttackConfig):
    pass

@dataclass
class LoktAttacker(PLGMIAttacker):
    pass