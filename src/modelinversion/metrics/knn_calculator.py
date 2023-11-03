from .basemetric import BaseMetricCalculator
import torch
from ..utils import Record
import os
import numpy as np

class KnnCalculator(BaseMetricCalculator):
    
    def __init__(self, model, recover_imgs_dir, real_imgs_dir, recover_feat_dir, real_feat_dir, batch_size=60, device='cpu') -> None:
        super().__init__(model, recover_imgs_dir, real_imgs_dir, recover_feat_dir, real_feat_dir, batch_size, True, device)
        
    def generate_feature(self):
        super().generate_feature(self.recover_feat_dir, self.get_recover_loader())
        super().generate_feature(self.real_feat_dir, self.get_real_loader)
        
    def calculate(self):
        fake_feat_dir = self.recover_feat_dir
        private_feat_dir = self.real_feat_dir
        
        fake_feat_files = os.listdir(fake_feat_dir)
    
        total_knn = 0
        total_num = 0
        
        print(f'calculate knn\n fake from {fake_feat_dir}\n private from {private_feat_dir}')
        
        for fake_feat_file in fake_feat_files:
            fake_path = os.path.join(fake_feat_dir, fake_feat_file)
            private_path = os.path.join(private_feat_dir, fake_feat_file)
            if not os.path.exists(private_path):
                continue
            
            # (N_f, 1, dim)
            fake_feat = np.load(fake_path)[:, None, :]
            # (1, N_p, dim)
            private_feat = np.load(private_path)[None, :, :]
            # (N_f, N_p)
            diff = ((fake_feat - private_feat) ** 2).sum(axis=-1)
            
            knns = np.min(diff, axis=1)
            total_knn += knns.sum()
            total_num += len(knns)
        if total_num == 0:
            raise RuntimeError('NO feat file for fake or private')
        
        return total_knn / total_num