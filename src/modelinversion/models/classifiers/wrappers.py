from ...utils import BaseHook
from .base import *

class TorchvisionClassifierModel(BaseImageClassifier):
    
    def __init__(self, arch_name: str, num_classes: int, resolution=224, weights=None, arch_kwargs={}, *args, **kwargs) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'
        
        super().__init__(resolution, feature_dim, num_classes)
        
        tv_module = importlib.import_module('torchvision.models')
        factory = getattr(tv_module, arch_name, None)
        model = factory(weights=weights, **arch_kwargs)
        
        self._feature_hook = None
        
        def _add_hook_fn(m):
            self._feature_hook = FirstInputHook(m)
        
        feature_dim = operate_fc(model, num_classes, _add_hook_fn)
        
        self.model = model
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)
    
    def get_last_feature_hook(self) -> BaseHook:
        return self._feature_hook
    
class VibWrapper(BaseImageClassifier):
    
    def __init__(self, module: BaseImageClassifier, *args, **kwargs) -> None:
        super().__init__(module.resolution, module.feature_dim, module.num_classes, *args, **kwargs)
        
        # assert module.feature_dim % 2 == 0
        
        self._inner_hook = module.get_last_feature_hook()
        
        if self._inner_hook is None:
            raise ModelConstructException('the module lack `last_feature_hook`')
        
        self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.fc_layer = nn.Linear(self.k, module.num_classes)
        
        self._last_statics = None, None
        
        self.feature_hook = FirstInputHook(self.fc_layer)
        
    @property
    def last_statics(self):
        return self._last_statics
    
    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook
        
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        
        self._inner_hook.clear_feature()
        self.module(image, *args, **kwargs)
        statis = self._inner_hook.get_feature()
        
        mu, std = statis[:, :self.k], statis[:, self.k: self.k * 2]
        
        self._last_statics = mu, std

        std = F.softplus(std - 5, beta=1)
        
        eps = torch.FloatTensor(std.size()).normal_().to(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)
        
        return out