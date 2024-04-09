import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import torch_mvn_logp
import numpy as np
from flow.model import Glow


def load_flow(inp_dim, hidden_channels, K, sn, nonlin, flow_permutation):
    glow_default = {
        'mlp': True,
        'image_shape': None,
        'actnorm_scale': 1,
        'flow_coupling': 'additive',
        'LU_decomposed': True,
        'y_classes': -1,
        'L': 0,  # Not used for MLP
        'learn_top': False,
        'y_condition': False,
        'logittransform': False,
        'use_binning_correction': False,
        'use_actnorm': False,
    }
    flow = Glow(
        inp_dim=inp_dim,
        hidden_channels=hidden_channels,
        K=K,
        sn=sn,
        nonlin=nonlin,
        flow_permutation=flow_permutation,
        **glow_default,
    )
    flow.return_ll_only = True
    return flow


def load_glow(
    inp_dim,
    hidden_channels,
    K,
    sn,
    nonlin,
    flow_permutation,
    flow_coupling,
    flow_L,
    use_actnorm,
):
    glow_default = {
        'mlp': False,
        'actnorm_scale': 1,
        'LU_decomposed': True,
        'y_classes': -1,
        'learn_top': False,
        'y_condition': False,
        'logittransform': False,
        'use_binning_correction': False,
    }
    flow = Glow(
        inp_dim=None,
        image_shape=(1, 1, inp_dim),
        hidden_channels=hidden_channels,
        K=K,
        sn=sn,
        nonlin=nonlin,
        flow_permutation=flow_permutation,
        flow_coupling=flow_coupling,
        L=flow_L,
        use_actnorm=use_actnorm,
        **glow_default,
    )
    flow.return_ll_only = True
    return flow


class FlowMiner(nn.Module):
    def __init__(
        self,
        nz0,
        flow_permutation,
        K,
        flow_glow=False,
        flow_coupling='additive',
        flow_L=1,
        flow_use_actnorm=True,
    ):
        super(FlowMiner, self).__init__()
        self.nz0 = nz0
        self.is_glow = flow_glow
        if flow_glow:
            self.flow = load_glow(
                inp_dim=self.nz0,
                hidden_channels=100,
                K=K,
                sn=False,
                nonlin='elu',
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                flow_L=flow_L,
                use_actnorm=flow_use_actnorm,
            )
            self.flow.cuda()
            # Init Actnorm
            init_z = torch.randn(100, self.nz0, 1, 1).cuda()
            self.flow(init_z)
        else:
            self.flow = load_flow(
                inp_dim=self.nz0,
                hidden_channels=100,
                K=K,
                sn=False,
                nonlin='elu',
                flow_permutation=flow_permutation,
            )

    def forward(self, z):
        if self.is_glow:
            z = z.unsqueeze(-1).unsqueeze(-1)
        z0 = self.flow.reverse_flow(z, y_onehot=None, temperature=1)
        if self.is_glow:
            z0 = z0.squeeze(-1).squeeze(-1)
        return z0

    def logp(self, x):
        if self.is_glow:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.flow(x)

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.flow.set_actnorm_init()


class LayeredFlowMiner(nn.Module):
    def __init__(
        self,
        k,
        l,
        flow_permutation,
        K,
        flow_glow=False,
        flow_coupling='additive',
        flow_L=1,
        flow_use_actnorm=True,
    ):
        """
        input
                k: num dim
                l: num component
        """
        super(LayeredFlowMiner, self).__init__()
        self.nz0 = k
        self.l = l
        self.flow_miners = [
            FlowMiner(
                self.nz0,
                flow_permutation,
                K,
                flow_glow,
                flow_coupling,
                flow_L,
                flow_use_actnorm,
            )
            for _ in range(self.l)
        ]
        for ll, flow_miner in enumerate(self.flow_miners):
            for name, p in flow_miner.named_parameters():
                name = name.replace('.', '_')
                self.register_parameter(f"_{ll}_{name}", p)

    def forward(self, z):
        z0s = [flow_miner(z) for flow_miner in self.flow_miners]
        z0s = torch.stack(z0s).permute(1, 0, 2)  # (N, l, nz0)
        return z0s

    def to(self, device):
        super(LayeredFlowMiner, self).to(device)
        for flow_miner in self.flow_miners:
            flow_miner.to(device)
        return self

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        for flow_miner in self.flow_miners:
            flow_miner.flow.set_actnorm_init()

    def eval(self):
        # super().eval()
        for flow_miner in self.flow_miners:
            flow_miner.flow.eval()

    def train(self):
        # super().train()
        for flow_miner in self.flow_miners:
            flow_miner.flow.train()


class MixtureOfRMVN(nn.Module):
    def __init__(self, k, l):
        """
        input
                k: num dim
                l: num component
        """
        super(MixtureOfRMVN, self).__init__()
        self.nz0 = k
        self.l = l
        self.mvns = [ReparameterizedMVN(self.nz0) for _ in range(self.l)]
        for ll, mvn in enumerate(self.mvns):
            for name, p in mvn.named_parameters():
                self.register_parameter(f"mvn_{ll}_{name}", p)

    def forward(self, z):
        z0s = [mvn(z) for mvn in self.mvns]
        z0s = torch.stack(z0s).permute(1, 0, 2)  # (N, l, nz0)
        return z0s


class MixtureOfIndependentRMVN(MixtureOfRMVN):
    def __init__(self, k, l):
        """
        input
                k: num dim
                l: num component
        """
        super(MixtureOfIndependentRMVN, self).__init__(k, l)

    def forward(self, zs):
        """
        input
            zs: tensor (num layers, batch size, dim)
        """
        assert len(zs) == len(self.mvns)
        z0s = [mvn(z) for (mvn, z) in zip(self.mvns, zs)]
        z0s = torch.stack(z0s).permute(1, 0, 2)  # (N, l, nz0)
        return z0s


# class ReparameterizedGMM(nn.Module):
#     def __init__(self, k, n_components):
#         super(ReparameterizedGMM, self).__init__()
#         self.nz0 = k
#         self.n_components = n_components
#         self.mvns = [ReparameterizedMVN(self.nz0) for _ in range(self.n_components)]
#         for ll, mvn in enumerate(self.mvns):
#             # Randomly Initialize the means
#             mvn.m.data = torch.randn_like(mvn.m.data)
#             # Register
#             for name, p in mvn.named_parameters():
#                 self.register_parameter(f"mvn_{ll}_{name}", p)
#         # self.mixing_weight_logits = nn.Parameter(torch.zeros(self.n_components))

#     # @property
#     # def mixing_weight(self):
#     #     return torch.softmax(self.mixing_weight_logits)

#     # def sample_components(self, n):
#     #     torch.distributions.Categorical(torch.from_numpy(np.array([0.1,0.9]))).sample((3,))

#     def forward(self, z):
#         batch_size = len(z)
#         # Sample components
#         inds = torch.randint(size=[batch_size], high=self.n_components)
#         masks = torch.eye(self.n_components)[inds]
#         masks = masks.t()  # (n_comps, batch_size)
#         masks = masks.to(z.device)

#         # Sample from all components
#         samps = torch.stack([mvn(z) for mvn in self.mvns])  # (n_comps, batch_size, ...)

#         # Selected Samples
#         x = (masks[...,None] * samps).sum(0)
#         return x

#     def logp(self, x):
#         logps = []
#         for mvn in self.mvns:
#             logp = mvn.logp(x)
#             logps.append(logp)
#         logps = torch.stack(logps)
#         logp = torch.mean(logps, 0)
#         return logp

#     def sample(self, N):
#         return self(torch.randn(N, self.nz0).to(self.m.device))


class MixtureOfGMM(nn.Module):
    def __init__(self, k, n_components, l):
        """
        input
                k: num dim
                l: num component
        """
        super(MixtureOfGMM, self).__init__()
        self.nz0 = k
        self.n_components = n_components
        self.l = l
        self.gmms = [
            ReparameterizedGMM2(self.nz0, self.n_components) for _ in range(self.l)
        ]
        for ll, gmm in enumerate(self.gmms):
            for name, p in gmm.named_parameters():
                self.register_parameter(f"gmm_{ll}_{name}", p)

    def forward(self, z):
        z0s = [gmm(z) for gmm in self.gmms]
        z0s = torch.stack(z0s).permute(1, 0, 2)  # (N, l, nz0)
        return z0s


class ReparameterizedGMM2(nn.Module):
    def __init__(self, k, n_components):
        super(ReparameterizedGMM2, self).__init__()
        self.nz0 = k
        self.n_components = n_components
        self.mvns = [ReparameterizedMVN(self.nz0) for _ in range(self.n_components)]
        for ll, mvn in enumerate(self.mvns):
            # Randomly Initialize the means
            mvn.m.data = torch.randn_like(mvn.m.data)
            # Register
            for name, p in mvn.named_parameters():
                self.register_parameter(f"mvn_{ll}_{name}", p)
        self.mixing_weight_logits = nn.Parameter(torch.zeros(self.n_components))

    def sample_components_onehot(self, n):
        return F.gumbel_softmax(self.mixing_weight_logits[None].repeat(n, 1), hard=True)

    def forward(self, z):
        batch_size = len(z)
        # Sample components
        masks = self.sample_components_onehot(batch_size)
        masks = masks.t()  # (n_comps, batch_size)

        # Sample from all components
        samps = torch.stack([mvn(z) for mvn in self.mvns])  # (n_comps, batch_size, ...)

        # Selected Samples
        x = (masks[..., None] * samps).sum(0)
        return x

    def logp(self, x):
        n = len(x)
        logps = []
        for mvn in self.mvns:
            logp = mvn.logp(x)
            logps.append(logp)
        logps = torch.stack(logps)  # (n_comp, n)
        log_mixing_weights = F.log_softmax(
            self.mixing_weight_logits[None].repeat(n, 1), dim=1
        ).t()
        logp = torch.logsumexp(logps + log_mixing_weights, dim=0) - np.log(
            self.n_components
        )
        return logp

    def sample(self, N):
        return self(torch.randn(N, self.nz0).to(self.m.device))


class ReparameterizedMVN(nn.Module):
    def __init__(self, k):
        super(ReparameterizedMVN, self).__init__()
        self.nz0 = k
        self.m = nn.Parameter(torch.zeros((1, k)).float())
        self.L = nn.Parameter(torch.eye(k).float())

    def forward(self, z):
        return self.m + z @ self.L.T

    def logp(self, x):
        C = self.L @ self.L.T
        return torch_mvn_logp(x, self.m, C)

    def entropy(self):
        C = self.L @ self.L.T
        H = (1 / 2) * torch.logdet(2 * np.pi * np.e * C)
        return H

    def sample(self, N):
        return self(torch.randn(N, self.nz0).to(self.m.device))


def test_mvn_opt():
    def plot_data_samples(data, samples, fname):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        plt.subplot(axs[0])
        plt.title("Data")
        plt.scatter(data.T[0], data.T[1])
        plt.subplot(axs[1])
        plt.title('Model')
        plt.scatter(samples.T[0], samples.T[1])
        plt.savefig(fname, bbox_inches='tight')

    # test logp
    m = torch.tensor([[2, 1]]).float()
    L = torch.tensor([[1, 2], [0, 1]]).float()
    C = L @ L.T
    gt_model = MultivariateNormal(m, covariance_matrix=C)
    X = gt_model.sample((5000,)).squeeze(1)

    model = ReparameterizedMVN(2)
    model.m.data = m
    model.L.data = L

    gt_logps = gt_model.log_prob(X)
    logps = model.logp(X)
    print(torch.sum(torch.abs(gt_logps - logps)))

    model = ReparameterizedMVN(2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    pbar = tqdm(range(0, 5000), desc='Train loop')
    for i in pbar:
        if i % 100 == 0:
            fname = f'likelihood_models_test/iter{i:4d}.jpeg'
            with torch.no_grad():
                samples = model(torch.randn(5000, 2))
            plot_data_samples(X, samples, fname)
        optimizer.zero_grad()
        loss = -model.logp(X).mean()
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(s=f'Loss: {loss.item():.2f}', refresh=True)


def test_mvn_entropy():
    model = ReparameterizedMVN(2)
    # test logp
    m = torch.tensor([[2, 1]]).float()
    L = torch.tensor([[1, 2], [0, 1]]).float()
    model.m.data = m
    model.L.data = L

    samples = model(torch.randn(5000, 2))
    H1 = -model.logp(samples).mean()
    H2 = model.entropy()
    print(H1, H2, H1 - H2)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = -model.entropy()
        loss.backward()
        optimizer.step()
    print(model.logp(samples).mean())


if __name__ == '__main__':
    from torch.distributions.multivariate_normal import MultivariateNormal
    import torch.optim as optim
    from tqdm import tqdm
    import matplotlib.pylab as plt
    from time import time

    # test_mvn_top()
    # test_mvn_entropy()
    # # Test Flow
    # flow = load_glow(hidden_channels=100,
    #           K=10,
    #           sn=False,
    #           nonlin='elu',
    #           flow_permutation='shuffle')
    # flow = flow.cuda()
    # z = torch.randn(100, 512, 1, 1).cuda()
    # lp = flow(z)
    # z1 = flow.reverse_flow(z, None, 1)

    # Test GMM
    N, D, C = 32, 512, 10
    gmm = ReparameterizedGMM2(D, C)
    noise = torch.randn(N, D)
    noise.requires_grad_()
    z = gmm(noise)
    start = time()
    lp = gmm.logp(z)
    end = time()
    print(end - start)

    start = time()
    lp.sum().backward()
    end = time()
    print(end - start)

    # path = '/scratch/hdd001/home/wangkuan/mm-icml2021/run_scripts/May19-celeba-dcgan-gmm-dev.sh-db0-1/expCelebA.1.DCGAN-m_gmm_ncomp5-lr1e-3-l-kl1e-3-id0/miner_10.pt'
    # sd = torch.load(path)

    import ipdb

    ipdb.set_trace()
