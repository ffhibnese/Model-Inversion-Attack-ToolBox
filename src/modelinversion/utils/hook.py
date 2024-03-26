from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.nn import Module, parallel



class BaseHook(metaclass=ABCMeta):
    """Monitor the model when forward
    """
    
    def __init__(self, module: Module) -> None:
        self.hook = module.register_forward_hook(self._hook_gather_impl)
        self.features = []
        
    def _hook_gather_impl(self, module, input, output):
        feature = self.hook_fn(module, input, output)
        self.features = feature #.append(feature)
        
    @abstractmethod
    def hook_fn(self, module, input, output):
        raise NotImplementedError()
    
    # def clear_feature(self) -> None:
    #     self.features.clear()
    
    def get_feature(self) -> Tensor:
        """
        Returns:
            Tensor: the value that the hook monitor.
        """
        return self.features
        # length = len(self.features)
        # if length == 0:
        #     return None
        # elif length == 1:
        #     return self.features[0]
        # return parallel.gather(self.features, target_device=target_device, dim=0) 
    
    def close(self):
        self.hook.remove()
        
class OutputHook(BaseHook):
    """Monitor the output of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        return output
        
class InputHook(BaseHook):
    """Monitor the input of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        return input
        
    
class FirstInputHook(BaseHook):
    """Monitor the input of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        return input[0]
        