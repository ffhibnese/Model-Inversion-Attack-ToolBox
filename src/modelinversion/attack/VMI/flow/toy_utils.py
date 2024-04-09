import numpy as np
from itertools import product


def twod_mog_means(n_centers):

    assert np.sqrt(n_centers) == np.floor(np.sqrt(n_centers))
    n_centers = int(np.sqrt(n_centers))
    std = 1.0 / n_centers
    means = np.arange(-3.0, 3.0, 6.0 / n_centers) + 6.0 / n_centers / 2
    means = np.array(list(product(means, means)))
    return means


def get_grid(low=-4, high=4, npts=20, ret_xy=False):
    delta = (high - low) / npts
    x, y = np.mgrid[low:high:delta, low:high:delta]
    # x, y = np.mgrid[low:high+delta:delta, low:high+delta:delta]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    if ret_xy:
        return x, y
    else:
        return pos.reshape(-1, 2)


def compute_grid_f(f, npts=100, low=-4.0, high=4.0):
    x = get_grid(low=low, high=high, npts=npts)
    fx = f(x)[:, None]
    return fx.reshape(npts, npts)


# def compute_density(logdensity, npts=100, low=-4., high=4.):
#     x = get_grid(low=low, high=high, npts=npts)
#     logpx = logdensity(x)[:, None]

#     px = np.exp(logpx).reshape(npts, npts)
#     # px = np.exp(logpx).reshape(npts+1, npts+1)
#     return px


# def plt_density(logdensity, ax, npts=100, low=-4., high=4., alpha=1, cmap='inferno'):
#     px = compute_grid_f(logdensity, npts, low, high)
#     ax.imshow(px, alpha=alpha, cmap=cmap)


def plt_contourf(f, ax, npts=20, low=-4.0, high=4.0, fill=True, **kwargs):
    px = compute_grid_f(f, npts, low, high)
    x, y = get_grid(low=low, high=high, npts=npts, ret_xy=True)
    if fill:
        cont = ax.contourf
    else:
        cont = ax.contour
    return cont(x, y, px, **kwargs)


def plt_samples(samples, ax, npts=100, low=-4.0, high=4.0, alpha=1, cmap='inferno'):
    ax.hist2d(
        samples[:, 0],
        samples[:, 1],
        range=[[low, high], [low, high]],
        bins=npts,
        alpha=alpha,
        cmap=cmap,
    )
    ax.invert_yaxis()
