from torch import nn, Tensor
from typing import Callable
from torch.nn import Module
from abc import abstractmethod, ABCMeta

def traverse_module(module: nn.Module, fn: Callable, call_middle=False):
    """Use DFS to traverse the module and visit submodules by function `fn`.

    Args:
        module (nn.Module): the module to be traversed
        fn (Callable): visit function
        call_middle (bool, optional): If true, it will visit both intermediate nodes and leaf nodes, else, it will only visit leaf nodes. Defaults to False.
    """

    children = list(module.children())
    if len(children) == 0:
        fn(module)
    else:
        if call_middle:
            fn(module)
        for child in children:
            traverse_module(child, fn)
            
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