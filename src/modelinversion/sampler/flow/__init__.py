from .likelihood_models import *
from dataclasses import dataclass, field


class LayeredMineGAN(nn.Module):
    def __init__(self, miner, Gmapping):
        super(LayeredMineGAN, self).__init__()
        self.nz = miner.nz0
        self.miner = miner
        self.Gmapping = Gmapping

    def forward(self, z0):
        N, zdim = z0.shape
        z = self.miner(z0)  # (N, zdim) -> (N, l, zdim)
        w = self.Gmapping(z.reshape(-1, zdim))  # (N * l, l, zdim)
        w = w[:, 0].reshape(N, -1, zdim)  # (N, l, zdim)
        return w


@dataclass
class FlowConfig:
    k: int
    l: int
    flow_permutation: str
    flow_K: int
    flow_glow: bool = False
    flow_coupling: str = 'additive'
    flow_L: int = 1
    flow_use_actnorm: bool = True
    l_identity: list = range(9)
