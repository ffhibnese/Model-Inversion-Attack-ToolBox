
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize

from .modelresult import ModelResult
from .base import BaseTargetModel
from ..utils import BaseHook, InputHook, OutputHook, traverse_module

class TorchVisionModelWrapper(BaseTargetModel):
    
    def __init__(self, model: nn.Module, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.resolution = 224
        
        leaf_modules = []
        traverse_module(self.model, lambda m: leaf_modules.append(m))
        self.fc_layer = None
        for m in leaf_modules[::-1]:
            if isinstance(m, nn.Linear):
                self.hook = InputHook(m)
                self.feat_dim = m.weight.shape[-1]
                if num_classes != m.weight.shape[0]:
                    self.fc_layer = nn.Linear(self.feat_dim, num_classes)
                break
            
    def get_feature_dim(self) -> int:
        return self.feat_dim
    
    def forward(self, x):
        
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = resize(x, [self.resolution, self.resolution])
            
        res = self.model(x)
        feature = self.hook.get_feature()[-1]
        if self.fc_layer is not None:
            res = self.fc_layer(feature)
            
        return  ModelResult(res, [feature])

# class TorchVisionModelWrapper(BaseTargetModel):
    
#     def __init__(self, model: nn.Module, num_classes):
#         super().__init__()
        
#         self.model = model
        
#         monitor_module = model
#         children = list(model.children())
#         while len(children) > 0:
#             monitor_module = children[-1]
#             children = list(monitor_module.children())
            
#         if not isinstance(monitor_module, nn.Linear):
#             raise NotImplementedError()  
        
#         self.hook = InputHook(monitor_module)
#         self.feat_dim = monitor_module.weight.shape[-1]
#         self.num_classes = num_classes
        
#         self.resolution = 224
        
#         self.flatten_models = []
#         traverse_module(self.model, lambda x: self.flatten_models.append(x))
        
#         # self.is_change_fc = False
#         # if self.num_classes != monitor_module.weight.shape[0]:
#         self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
#             # self.is_change_fc = True
            
        
#     def get_feature_dim(self) -> int:
#         return self.feat_dim
    
#     def create_hidden_hooks(self) -> list:
        
#         hiddens_hooks = []
        
#         length_hidden = len(self.flatten_models)
        
#         num_body_monitor = 4
#         offset = length_hidden // num_body_monitor
#         for i in range(num_body_monitor):
#             hiddens_hooks.append(OutputHook(self.flatten_models[offset * (i+1) - 1]))
        
#         # hiddens_hooks.append(OutputHook(self.output_layer))
#         return hiddens_hooks
    
#     def freeze_front_layers(self) -> None:
#         length_hidden = len(self.flatten_models)
#         for i in range(int(length_hidden * 2 // 3)):
#             self.flatten_models[i].requires_grad_(False)
            
#     def forward(self, x):
        
#         if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
#             x = resize(x, [self.resolution, self.resolution])
            
#         res = self.model(x)
#         # feature = feature.view(feature.size(0), -1)
#         feature = self.hook.get_feature()[0]
        
#         # if self.is_change_fc:
#         res = self.fc_layer(feature)
#         return  ModelResult(res, [feature])

class VibWrapper(nn.Module):
    
    def __init__(self, module: nn.Module, hidden_dim: int, output_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert hidden_dim % 2 == 0
        
        self.module = module
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = self.hidden_dim // 2
        self.fc_layer = nn.Linear(self.k, output_dim)
        
    def forward(self, *args, **kwargs):
        statis = self.module(*args, **kwargs).feat[-1]
        
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        
        eps = torch.FloatTensor(std.size()).normal_().to(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)

        output = ModelResult(out, [feat], {'mu': mu, 'std': std})
        return output
    