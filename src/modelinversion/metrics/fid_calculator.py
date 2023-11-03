from .basemetric import BaseMetricCalculator
import torch
from ..utils import Record
import os
import numpy as np
import collections
import torch.nn.functional as F
from scipy import linalg

class FidCalculator(BaseMetricCalculator):
    
    def __init__(self, model, recover_imgs_dir, real_imgs_dir, recover_feat_dir, real_feat_dir, batch_size=60, device='cpu') -> None:
        super().__init__(model, recover_imgs_dir, real_imgs_dir, recover_feat_dir, real_feat_dir, batch_size, False, device)
        
    def _generate_feature(self, save_dir, dataloader):
        self.model.eval()
        with torch.no_grad():
            results = []
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                pred = self.model(imgs).feat[-1]
                
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                results.append(pred.detach().cpu().numpy().reshape(len(imgs), -1))
            results = np.concatenate(results, axis=0)
            
        mu = np.mean(results, axis=0)
        var = np.cov(results, rowvar=False)
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'fid_mu.npy'), mu)
        np.save(os.path.join(save_dir, 'fid_sigma.npy'), var)
            
        
    def generate_feature(self):
        self._generate_feature(self.recover_feat_dir, self.get_recover_loader())
        self._generate_feature(self.real_feat_dir, self.get_real_loader)
        
    def calculate(self):
        mu1 = np.load(os.path.join(self.recover_feat_dir, 'fid_mu.npy'))
        mu2 = np.load(os.path.join(self.real_feat_dir, 'fid_mu.npy'))
        sigma1 = np.load(os.path.join(self.recover_feat_dir, 'fid_sigma.npy'))
        sigma2 = np.load(os.path.join(self.real_feat_dir, 'fid_sigma.npy'))
        
        fid_res = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        print(f'fid: {fid_res:.6f}')
        
        return fid_res
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean