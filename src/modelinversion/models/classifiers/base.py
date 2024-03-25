import importlib
from abc import abstractmethod
from typing import Callable, Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodel
import torchvision.transforms.functional as TF

from ...utils import traverse_name_module, FirstInputHook, BaseHook

class ModelConstructException(Exception):
    pass

class BaseTargetModel(nn.Module):
    
    pass
    
class BaseImageModel(BaseTargetModel):
    
    def __init__(self, resolution: int, feature_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._resolution = resolution
        self._feature_dim = feature_dim
        
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def feature_dim(self):
        return self._feature_dim
    
    @abstractmethod
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()
    
    def forward(self, image: torch.Tensor, *args, **kwargs):
        
        if image.shape[-1] != self.resolution or image.shape[-2] != self.resolution:
            image = TF.resize(image, (self.resolution, self.resolution), antialias=True)
            
        return self._forward_impl(image, *args, **kwargs)
    
class BaseImageEncoder(BaseImageModel):
    
    def __init__(self, resolution: int, feature_dim: int, *args, **kwargs) -> None:
        super().__init__(resolution, feature_dim, *args, **kwargs)
    
class BaseImageClassifier(BaseImageModel):
    
    def __init__(self, resolution, feature_dim, num_classes, *args, **kwargs) -> None:
        super().__init__(resolution, feature_dim, *args, **kwargs)
        self._num_classes = num_classes
        
    @property
    def num_classes(self):
        return self._num_classes
    
    def get_last_feature_hook(self) -> BaseHook:
        return None

    
def _operate_fc_impl(module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None):
    
    if isinstance(module, nn.Sequential):
        
        if len(module) == 0:
            raise ModelConstructException('fail to implement')
        
        if isinstance(module[-1], nn.Linear):
            feature_dim = module[-1].weight.shape[-1]
            
            if reset_num_classes is not None and reset_num_classes != module[-1].weight.shape[0]:
                module[-1] = nn.Linear(feature_dim, reset_num_classes)
                
            if visit_fc_fn is not None:
                visit_fc_fn(module[-1])
                
            return feature_dim
        else:
            return _operate_fc_impl(module[-1], reset_num_classes)
    
    children = list(module.named_children())
    if len(children) == 0:
        raise ModelConstructException('fail to implement')
    attr_name, child_module = children[-1]
    if isinstance(child_module, nn.Linear):
        feature_dim = child_module.weight.shape[-1]
        
        if reset_num_classes is not None and reset_num_classes != child_module.weight.shape[0]:
            setattr(module, attr_name, nn.Linear(feature_dim, reset_num_classes))
            
        if visit_fc_fn is not None:
            visit_fc_fn(getattr(module, attr_name))
            
        return feature_dim
    else:
        return _operate_fc_impl(child_module, reset_num_classes)
        
    
def operate_fc(module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None) -> int:
    return _operate_fc_impl(module, reset_num_classes, visit_fc_fn)
    
    
