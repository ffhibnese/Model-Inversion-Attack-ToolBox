from torch import Tensor
from ...utils import BaseHook
from .base import *

class TorchvisionClassifierModel(BaseImageClassifier):
    
    def __init__(self, arch_name: str, num_classes: int, resolution=224, weights=None, arch_kwargs={}, register_last_feature_hook=False, *args, **kwargs) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'
        
        self._feature_hook = None
        
        def _add_hook_fn(m):
            self._feature_hook = FirstInputHook(m)
            
        tv_module = importlib.import_module('torchvision.models')
        factory = getattr(tv_module, arch_name, None)
        model = factory(weights=weights, **arch_kwargs)
        
        feature_dim = operate_fc(model, num_classes, _add_hook_fn)
        
        super().__init__(resolution, feature_dim, num_classes, register_last_feature_hook)
        
        self.model = model
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)
    
    def get_last_feature_hook(self) -> BaseHook:
        return self._feature_hook
    
class VibWrapper(BaseImageClassifier):
    
    def __init__(self, module: BaseImageClassifier, register_last_feature_hook=False, *args, **kwargs) -> None:
        super().__init__(module.resolution, module.feature_dim, module.num_classes, register_last_feature_hook, *args, **kwargs)
        
        self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.fc_layer = nn.Linear(self.k, module.num_classes)
        
        self.feature_hook = FirstInputHook(self.fc_layer)
        
    
    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook
        
        
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        
        # self._inner_hook.clear_feature()
        _, hook_res = self.module(image, *args, **kwargs)
        
        self._check_hook(HOOK_NAME_FEATURE)
        
        statis = hook_res[HOOK_NAME_FEATURE]
        
        mu, std = statis[:, :self.k], statis[:, self.k: self.k * 2]
        
        self._last_statics = mu, std

        std = F.softplus(std - 5, beta=1)
        
        # eps = torch.FloatTensor(std.size()).normal_().to(std)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)
        
        return out, {'mu': mu, 'std': std}
    
def get_default_create_hidden_hook_fn(num: int = 3):
    
    def _fn(model: BaseImageClassifier):
        linear_modules = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                linear_modules.append(m)
        linear_modules = linear_modules[1:]
                
        num = min(num, len(linear_modules))
        splitnum = (len(linear_modules)+1) // (num + 1)
        use_nums = [splitnum * (i+1) - 1 for i in range(num)]
        use_linear_modules = [linear_modules[i] for i in use_nums]
        return [FirstInputHook(l) for l in use_linear_modules]
    return _fn
    
class BiDOWrapper(BaseImageClassifier):
    
    def __init__(self, module: BaseImageClassifier, register_last_feature_hook=False, create_hidden_hook_fn: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__(module.resolution, module.feature_dim, module.num_classes, register_last_feature_hook, *args, **kwargs)
        
        self.module = module
        
        create_hidden_hook_fn = create_hidden_hook_fn if create_hidden_hook_fn is not None else get_default_create_hidden_hook_fn()
        
        self.hidden_hooks = create_hidden_hook_fn(module)
        
    def get_last_feature_hook(self) -> BaseHook:
        return self.module.get_last_feature_hook()
    
    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        return forward_res, addition_info