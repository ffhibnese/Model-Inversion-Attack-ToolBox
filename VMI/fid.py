from tqdm import tqdm
import torch
import numpy as np
from scipy import linalg
# FID
import sys
sys.path.append('../stylegan2-ada-pytorch')
from metrics import metric_utils


device = 'cuda:0'
_feature_detector_cache = None
def get_feature_detector():
    global _feature_detector_cache
    if _feature_detector_cache is None:
        _feature_detector_cache = metric_utils.get_feature_detector(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/'
            'metrics/inception-2015-12-05.pt', device)
    return _feature_detector_cache


def postprocess(x):
    """."""
    return ((x * .5 + .5) * 255).to(torch.uint8)


def run_fid(x1, x2):
    # Extract features
    x1 = run_batch_extract(x1, device)
    x2 = run_batch_extract(x2, device)

    npx1 = x1.detach().cpu().numpy()
    npx2 = x2.detach().cpu().numpy()
    mu1 = np.mean(npx1, axis=0)
    sigma1 = np.cov(npx1, rowvar=False)
    mu2 = np.mean(npx2, axis=0)
    sigma2 = np.cov(npx2, rowvar=False)
    frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return frechet


def run_feature_extractor(x):
    assert x.dtype == torch.uint8
    assert x.min() >= 0
    assert x.max() <= 255
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    feature_extractor = get_feature_detector()
    return feature_extractor(x, return_features=True)


def run_batch_extract(x, device, bs=500):
    z = []
    with torch.no_grad():
        for start in tqdm(range(0, len(x), bs), desc='run_batch_extract'):
            stop = start + bs
            x_ = x[start:stop].to(device)
            z_ = run_feature_extractor(postprocess(x_)).cpu()
            z.append(z_)
    z = torch.cat(z)
    return z


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, return_details=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    if not return_details:
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    else:
        t1 = diff.dot(diff)
        t2 = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return (t1 + t2), t1, t2

if __name__ == '__main__':
    # Load Data
    target_x, target_y = torch.load('celeba_target_100ids.pt')

    # Load Samples
    fake = torch.load('results/images_pt/original_im.pt')

    # FID
    fid = run_fid(target_x, fake)
    print(f"Original:{fid}")

    # Load Independent Samples
    fake = torch.load('results/images_pt/independent_im.pt')

    # FID
    fid = run_fid(target_x, fake)
    print(f"Independent:{fid}")

