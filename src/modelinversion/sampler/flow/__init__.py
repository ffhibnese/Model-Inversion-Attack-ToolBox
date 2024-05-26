from .likelihood_models import *

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