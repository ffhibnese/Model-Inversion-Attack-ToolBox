import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np
from numpy import ndarray
from scipy import linalg
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ..foldermanager import FolderManager
from ..models.base import BaseTargetModel

class BaseMetric(ABC):
    
    def __init__(self, folder_manager: FolderManager, device:str) -> None:
        self.folder_manager = folder_manager
        self.device = device
        self.metric_name = self.get_metric_name()
    
    @abstractmethod
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> Any:
        raise NotImplementedError()
    
    @abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError()
    
class LabelSpecMetric(BaseMetric):
    
    def __init__(self, folder_manager: FolderManager, device:str) -> None:
        super().__init__(folder_manager, device)
        
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> Any:
        # step 1: generate features
        
        tag = f'{tag}.pt'
        
        metric_folder = os.path.join(self.folder_manager.config.cache_dir, self.metric_name, self.metric_name)
        os.makedirs(metric_folder, exist_ok=True)
        
        private_save_path = os.path.join(metric_folder, 'private.pt')
        
        # if not os.path.exists(private_save_path):
        #     print(f'generate features of real images')
        #     self.save_features(private_path, private_save_path, batch_size, transform=transform)
        
    
        # invert_save_path = os.path.join(metric_folder, tag)
        # print(f'generate features of inverted images')
        # self.save_features(invert_path, invert_save_path, batch_size, transform)
        
        
            
        # step 2: calculation
        # invert_features = np.load(invert_save_path)
        # private_features = np.load(private_save_path)
        
        # result = self.get_final_result(invert_features, private_features)
        # print(f'{self.metric_name}: {result}')
        # return result
    
class GeneralMetric(BaseMetric):
    
    def __init__(self, folder_manager: FolderManager, device:str) -> None:
        super().__init__(folder_manager, device)
        
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> Any:
        
        # step 1: generate features
        
        tag = f'{tag}.pt'
        
        metric_folder = os.path.join(self.folder_manager.config.cache_dir, self.metric_name, self.metric_name)
        os.makedirs(metric_folder, exist_ok=True)
        
        private_save_path = os.path.join(metric_folder, 'private.pt')
        
        if not os.path.exists(private_save_path):
            print(f'generate features of real images')
            self.save_features(private_path, private_save_path, batch_size, transform=transform)
        
    
        invert_save_path = os.path.join(metric_folder, tag)
        print(f'generate features of inverted images')
        self.save_features(invert_path, invert_save_path, batch_size, transform)
        
        
            
        # step 2: calculation
        invert_features = np.load(invert_save_path)
        private_features = np.load(private_save_path)
        
        result = self.get_final_result(invert_features, private_features)
        print(f'{self.metric_name}: {result}')
        return result
        
        
    def save_features(self, img_path, save_path, batch_size, transform=None):
        ds = ImageFolder(img_path, transform=transform)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            inputs = []
            for img, _ in dataloader:
                img = img.to(self.device)
                feat = self.input_transform(img)
                inputs.append(feat)
            inputs = np.concatenate(inputs, axis=0)
        
        features = self.generate_features(inputs)
        # np.save(save_path, features)
        torch.save(features, save_path)
        
    
    @abstractmethod
    def input_transform(self, img_tensor: Tensor) -> ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def generate_features(self, inputs: list) -> ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def get_final_result(self, invert_features, private_features) -> float:
        raise NotImplementedError()
    
class FIDMetric(GeneralMetric):
    
    def __init__(self, folder_manager: FolderManager, device: str, model: BaseTargetModel) -> None:
        super().__init__(folder_manager, device)
        self.model = model
        
    def get_metric_name(self) -> str:
        return 'fid'
    
    def input_transform(self, img_tensor: Tensor) -> ndarray:
        features = self.model(img_tensor).feat[-1]
        
        if features.shape[2] != 1 or features.shape[3] != 1:
            features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
            
        return features.cpu().numpy()
    
    def generate_features(self, inputs: list) -> ndarray:
        mu = np.mean(inputs, axis=0)
        var = np.cov(inputs, rowvar=False)
        
        return {'mu': mu, 'var': var}
    
    def get_final_result(self, invert_features, private_features) -> float:
        
        mu1, mu2 = invert_features['mu'], private_features['mu']
        sigma1, sigma2 = invert_features['var'], private_features['var']
        
        eps = 1e-6
        
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