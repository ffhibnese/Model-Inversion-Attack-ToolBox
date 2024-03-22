import importlib
from abc import abstractmethod
from typing import Callable, Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodel
import torchvision.transforms.functional as TF

from ...utils import traverse_name_module, InputHook, BaseHook

class ModelConstructException(Exception):
    pass

class BaseTargetModel(nn.Module):
    
    pass
    
class BaseImageModel(BaseTargetModel):
    
    def __init__(self, resolution, feature_dim, *args, **kwargs) -> None:
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
    
class BaseImageClassifier(BaseImageModel):
    
    def __init__(self, resolution, feature_dim, num_classes, last_feature_hook: Optional[BaseHook]=None, *args, **kwargs) -> None:
        super().__init__(resolution, feature_dim, *args, **kwargs)
        self._num_classes = num_classes
        self._last_feature_hook = last_feature_hook
        
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def last_feature_hook(self):
        return self._last_feature_hook

    
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
    
    
class TorchvisionClassifierModel(BaseImageClassifier):
    
    def __init__(self, arch_name: str, num_classes: int, weights=None, arch_kwargs={}, *args, **kwargs) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'
        
        tv_module = importlib.import_module('torchvision.models')
        factory = getattr(tv_module, arch_name, None)
        model = factory(weights=weights, **arch_kwargs)
        
        self.__feature_hook = None
        
        def _add_hook_fn(m):
            self.__feature_hook = InputHook(m)
        
        feature_dim = operate_fc(model, num_classes, _add_hook_fn)
        
        super().__init__(224, feature_dim, num_classes, self.__feature_hook)
        
        self.model = model
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)
    
class VibWrapper(BaseImageClassifier):
    
    def __init__(self, module: BaseImageClassifier, *args, **kwargs) -> None:
        super().__init__(module.resolution, module.feature_dim, module.num_classes, module.last_feature_hook, *args, **kwargs)
        
        # assert module.feature_dim % 2 == 0
        
        self._inner_hook = module.last_feature_hook
        
        if module.last_feature_hook is None:
            raise ModelConstructException('the module lack `last_feature_hook`')
        
        self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.fc_layer = nn.Linear(self.k, module.num_classes)
        
        self._last_statics = None, None
        
    @property
    def last_statics(self):
        return self._last_statics
        
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        self.module(image, *args, **kwargs)
        
        statis = self._inner_hook.get_feature()
        
        mu, std = statis[:, :self.k], statis[:, self.k: self.k * 2]
        
        self._last_statics = mu, std

        std = F.softplus(std - 5, beta=1)
        
        eps = torch.FloatTensor(std.size()).normal_().to(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)
        
        return out