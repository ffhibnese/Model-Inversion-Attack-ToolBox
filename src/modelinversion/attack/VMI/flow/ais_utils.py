import torch
import numpy as np
import torch.nn as nn

log2pi = float(np.log(2 * np.pi))

## Toy Dataset


## Toy Model
class Generator(object):
    def __init__(self, input_dim, output_dim, mean, scale):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean = mean
        self.std = scale

    def __call__(self, z):
        return z * self.std + self.mean


class MOGGenerator(object):

    def __init__(self, means, logsds):
        raise  #
        """
        This gives problem because it's stochastic
        So the same z can map to different mixture everytime
        This stochasticity (when taken into account) breaks AIS
        """
        self.n_components = len(means)
        self.means = means
        self.logsds = logsds

    def __call__(self, z):
        batch_size = z.shape[0]
        inds = torch.eye(self.n_components)[
            torch.randint(size=[batch_size], high=self.n_components)
        ]
        inds = inds.t().unsqueeze(-1)  # (n_comps, batch_size)
        samps = z.unsqueeze(0).repeat(
            self.n_components, 1, 1
        )  # (n_comps, batch_size, ...)
        return (
            inds
            * (samps * torch.exp(self.logsds.unsqueeze(1)) + self.means.unsqueeze(1))
        ).sum(0)


class MOGGenerator2(object):
    """
    a deterministic generator... not entirely satisfying tho
    TODO(jackson):
        how does AIS handle this, when model has extra stochasticity?
        Maybe it cannot......

    """

    def __init__(self, means, logsds):
        self.n_components = len(means)
        self.means = means
        self.logsds = logsds

    def __call__(self, z):
        batch_size = z.shape[0]
        _inds = torch.range(0, batch_size - 1) % self.n_components
        inds = torch.eye(self.n_components)[_inds.long()]
        inds = inds.t().unsqueeze(-1)  # (n_comps, batch_size)
        samps = z.unsqueeze(0).repeat(
            self.n_components, 1, 1
        )  # (n_comps, batch_size, ...)
        return (
            inds
            * (samps * torch.exp(self.logsds.unsqueeze(1)) + self.means.unsqueeze(1))
        ).sum(0)


## Observation Model
def kde_logpdf(x, mu, std):
    """
    Calculate the kde logpdf.
    Input:
        x - Shape [N, k]
        mu - Shape [N, k]
        std - standard devaition (1, )
    Return
         [N]
    """
    k = mu.size(1)
    #
    d = x.float() - mu.float()
    log_maha_dist = -0.5 * torch.pow(d, 2).sum(-1) / std**2

    neg_log_z = 0.5 * k * log2pi
    neg_log_z += (k * torch.log(std)).float()

    return log_maha_dist - neg_log_z


## AIS tools
def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return ((s - np.min(s)) / (np.max(s) - np.min(s))).astype('float32')


## Misc
def tensor(stuff):
    return torch.FloatTensor(np.array(stuff))


def log_mean_exp(x, axis=None):
    assert axis is not None
    # m = torch.max(x, dim=axis, keepdim=True)[0]
    # return m + torch.log(torch.mean(
    #     torch.exp(x - m), dim=axis, keepdim=True))
    return torch.logsumexp(x, dim=axis).squeeze() - torch.log(
        torch.ones(1).to(x.device) * x.size(axis)
    )


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps


def standard_gaussian(shape):
    mean, logsd = [torch.FloatTensor(*shape).fill_(0.0) for _ in range(2)]
    return gaussian_diag(mean, logsd)


class gaussian_diag(nn.Module):
    def __init__(self, mean, logsd, requires_grad=False):
        super(gaussian_diag, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=requires_grad)
        self.logsd = nn.Parameter(logsd, requires_grad=requires_grad)

    def logps(self, x):
        return -0.5 * (
            log2pi
            + 2.0 * self.logsd
            + ((x - self.mean) ** 2) / torch.exp(2.0 * self.logsd)
        )

    def logp(self, x):
        return flatten_sum(self.logps(x))

    def sample(self, batch_size):
        mean, logsd = self.mean, self.logsd
        shape = [batch_size] + list(mean.shape)
        eps = torch.zeros(shape).normal_().to(mean.device)
        return mean[None] + torch.exp(logsd[None]) * eps


class MOG(object):
    def __init__(self, means, logsds, requires_grad=False):
        # assumes uniform weight
        assert len(means) == len(logsds)  # n_components
        self.n_components = len(means)
        self.n_classes = len(means)
        self.means = means
        self.logsds = logsds
        self.gs = []
        for idx in range(len(means)):
            self.gs.append(gaussian_diag(means[idx], logsds[idx], requires_grad))

    def logp(self, sample):
        return log_mean_exp(torch.stack([g.logp(sample) for g in self.gs]), 0)

    def sample(self, batch_size):
        inds = torch.eye(self.n_components)[
            torch.randint(size=[batch_size], high=self.n_components)
        ]
        inds = inds.t()  # (n_comps, batch_size)
        samps = torch.stack(
            [g.sample(batch_size) for g in self.gs]
        )  # (n_comps, batch_size, ...)
        for _ in range(len(samps.shape) - 2):
            inds = inds.unsqueeze(-1)
        return (inds * samps).sum(0)

    def logpyx(self, sample):
        log_pjoint = torch.stack([g.logp(sample) for g in self.gs]).transpose(
            1, 0
        ) - np.log(self.n_components)
        log_px = torch.logsumexp(log_pjoint, 1, keepdim=True)
        log_pyx = log_pjoint - log_px
        assert log_pjoint.size(0) == sample.size(0)
        return log_pyx

    def to(self, device):
        newgs = []
        for g in self.gs:
            newgs.append(g.to(device))
        self.gs = newgs
