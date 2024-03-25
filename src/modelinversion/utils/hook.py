from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.nn import Module


class BaseHook(metaclass=ABCMeta):
    """Monitor the model when forward
    """
    
    def __init__(self, module: Module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)
        
    @abstractmethod
    def hook_fn(self, module, input, output):
        raise NotImplementedError()
    
    @abstractmethod
    def get_feature(self) -> Tensor:
        """
        Returns:
            Tensor: the value that the hook monitor.
        """
        raise NotImplementedError()
    
    def close(self):
        self.hook.remove()
        
class OutputHook(BaseHook):
    """Monitor the output of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        self.feature = output
        
    def get_feature(self):
        return self.feature
    
class InputHook(BaseHook):
    """Monitor the input of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        self.feature = input
        
    def get_feature(self):
        return self.feature
    
class FirstInputHook(BaseHook):
    """Monitor the input of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        self.feature = input[0]
        
    def get_feature(self):
        return self.feature