import os
import warnings
from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict

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
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError()
    
class LabelSpecMetric(BaseMetric):
    
    def __init__(self, folder_manager: FolderManager, device:str) -> None:
        super().__init__(folder_manager, device)
        
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> float:
        # step 1: generate features
        
        tag = f'{tag}.pt'
        
        metric_folder = os.path.join(self.folder_manager.config.cache_dir, self.metric_name, self.metric_name)
        
        
        private_save_dir = os.path.join(metric_folder, 'private')
        os.makedirs(metric_folder, exist_ok=True)
        
        class_len = len(os.listdir(private_path))
        if len(os.listdir(private_save_dir)) < class_len:
            print(f'generate features of private images')
            self.save_features(private_path, private_save_dir, batch_size, transform=transform)
        
        print(f'generate features of private images')
        invert_save_dir = os.path.join(metric_folder, tag)
        os.makedirs(metric_folder, exist_ok=True)
        self.save_features(invert_path, invert_save_dir, batch_size, transform=transform)
        
        # step 2: calculate result for each label
        
        score = 0
        total_num = 0
        
        for filename in os.listdir(invert_save_dir):
            invert_file = os.path.join(invert_save_dir, filename)
            private_file = os.path.join(private_save_dir, filename)
            if not os.path.exists(private_file):
                class_id = filename[:filename.rfind('.')]
                warnings.warn(f'class {class_id} is not existed in private data')
                continue
            invert_features = torch.load(invert_file)
            private_features = torch.load(private_file)
            num = len(invert_features)
            total_num += num
            score += num * self.get_label_final_result(invert_features, private_features)
        
        if total_num == 0:
            warnings.warn(f'no feature of inverted or private images')
            return 0
        
        result = score / total_num
        print(f'{self.metric_name}: {result}')
        return result
    
    def save_features(self, img_path, save_dir, batch_size, transform=None):
        
        ds = ImageFolder(img_path, transform=transform)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        label_dict = defaultdict(list)
        
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                feats = self.input_transform(imgs)
                labels = labels.numpy()
                for feat, label in zip(feats, labels):
                    label_dict[label].append(feat[np.newaxis,:])
                # for i in range(len())
                    
        for label, values in label_dict.items():
            values = np.concatenate(values)
            save_path = os.path.join(save_dir, f'{label}.pt')
            torch.save(values, save_path)
            
    @abstractmethod
    def input_transform(self, img_tensor: Tensor) -> ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def get_label_final_result(self, invert_features, private_features) -> float:
        raise NotImplementedError()
  
class KnnDistanceMetric(LabelSpecMetric):
    
    def __init__(self, folder_manager: FolderManager, device: str, model: BaseTargetModel) -> None:
        super().__init__(folder_manager, device)
        self.model = model
        
    def get_metric_name(self) -> str:
        return 'knn'
    
    def input_transform(self, img_tensor: Tensor) -> ndarray:
        bs = len(img_tensor)
        features = self.model(img_tensor).feat[-1].reshape(bs, -1)
        return features.cpu().numpy()
    
    def get_label_final_result(self, invert_features, private_features) -> float:
        # ( N_i, 1, dim )
        invert_features = invert_features[:, np.newaxis, :]
        # ( 1, N_p, dim )
        private_features = private_features[np.newaxis, :, :]
        
        diff = ((invert_features - private_features) ** 2).sum(axis=-1)
        knns = np.min(diff, axis=1)
        return knns.mean()
    

class FeatureDistanceMetric(LabelSpecMetric):
    
    def __init__(self, folder_manager: FolderManager, device: str, model: BaseTargetModel) -> None:
        super().__init__(folder_manager, device)
        self.model = model
        
    def get_metric_name(self) -> str:
        return 'feature_dist'
    
    def input_transform(self, img_tensor: Tensor) -> ndarray:
        bs = len(img_tensor)
        features = self.model(img_tensor).feat[-1].reshape(bs, -1)
        return features.cpu().numpy()
    
    def get_label_final_result(self, invert_features, private_features) -> float:

        private_features = np.mean(private_features, axis=0, keepdims=True)
        
        diff = ((invert_features - private_features) ** 2).sum(axis=-1)
        return diff.mean()     
 
class GeneralMetric(BaseMetric):
    
    def __init__(self, folder_manager: FolderManager, device:str) -> None:
        super().__init__(folder_manager, device)
        
    def calculate(self, tag: str, invert_path, private_path, batch_size, transform=None, **kwargs) -> float:
        
        # step 1: generate features
        
        tag = f'{tag}.pt'
        
        metric_folder = os.path.join(self.folder_manager.config.cache_dir, self.metric_name, self.metric_name)
        os.makedirs(metric_folder, exist_ok=True)
        
        private_save_path = os.path.join(metric_folder, 'private.pt')
        
        if not os.path.exists(private_save_path):
            print(f'generate features of private images')
            self.save_features(private_path, private_save_path, batch_size, transform=transform)
        
    
        invert_save_path = os.path.join(metric_folder, tag)
        print(f'generate features of inverted images')
        self.save_features(invert_path, invert_save_path, batch_size, transform)
        
        
            
        # step 2: calculation
        invert_features = torch.load(invert_save_path)
        private_features = torch.load(private_save_path)
        
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
        
        if len(mu1) <= 2048:
            warnings.warn(f'calculate fid failed: the number of inverted images is less than 2048')
            return 0
        
        if len(mu2) <= 2048:
            warnings.warn(f'calculate fid failed: the number of private images is less than 2048')
            return 0
        
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