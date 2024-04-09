import torch
import numpy as np


def distmat(X):
    """distance matrix"""
    assert X.ndim == 2
    r = torch.sum(X * X, dim=1, keepdim=True)
    # r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def hisc_kernelmat(X, sigma, ktype='gaussian'):
    """kernel matrix baker"""
    m = int(X.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2.0 * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError(
                    "Unstable sigma {} with maximum/minimum input ({},{})".format(
                        sx, torch.max(X), torch.min(X)
                    )
                )

    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = hisc_kernelmat(x, sigma=sigma)
    Kyc = hisc_kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1e-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def hsic_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    hsic_hx_val = hsic_normalized_cca(hidden, h_data, sigma=sigma)
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return hsic_hx_val, hsic_hy_val


def coco_kernelmat(X, sigma, ktype='gaussian'):
    """kernel matrix baker"""
    m = int(X.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2.0 * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError(
                    "Unstable sigma {} with maximum/minimum input ({},{})".format(
                        sx, torch.max(X), torch.min(X)
                    )
                )

    ## Adding linear kernel
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(H, torch.mm(Kx, H))

    return Kxc


def coco_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    K = coco_kernelmat(x, sigma=sigma)
    L = coco_kernelmat(y, sigma=sigma, ktype=ktype)

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res


def coco_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    coco_hx_val = coco_normalized_cca(hidden, h_data, sigma=sigma)
    coco_hy_val = coco_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return coco_hx_val, coco_hy_val
