import inspect



import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize

from .modelresult import ModelResult
from .base import BaseTargetModel
from ..utils import BaseHook, InputHook, OutputHook, traverse_name_module

class TorchVisionModelWrapper(BaseTargetModel):
    
    def __init__(self, model: nn.Module, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.resolution = 224
        
        leaf_modules = []
        traverse_name_module(self.model, lambda m: leaf_modules.append(m))
        for m_tuple in leaf_modules[::-1]:
            attr_name, m = m_tuple
            if isinstance(m, nn.Linear):
                self.feat_dim = m.weight.shape[-1]
                if num_classes != m.weight.shape[0]:
                    print('change fc')
                    # self.fc_layer = nn.Linear(self.feat_dim, num_classes)
                    # self.model.fc = nn.Linear(self.feat_dim, self.num_classes)
                    setattr(self.model, attr_name, nn.Linear(self.feat_dim, self.num_classes))
                self.hook = InputHook(getattr(self.model, attr_name))
                break
            
    def get_feature_dim(self) -> int:
        return self.feat_dim
    
    def forward(self, x):
        
#         if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
#             x = resize(x, [self.resolution, self.resolution])
            
        res = self.model(x)
        feature = self.hook.get_feature()[-1]
        return  ModelResult(res, [feature])

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
    