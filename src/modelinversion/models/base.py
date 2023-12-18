
from abc import abstractmethod

from torch import nn

class BaseTargetModel(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Returns:
            int: feature dim
        """
        raise NotImplementedError()
    
    @abstractmethod
    def create_hidden_hooks(self) -> list:
        """
        Create hooks for hidden features. BiDO defense method will use it.

        Returns:
            list: list of hidden features.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def freeze_front_layers(self) -> None:
        """
        The forward process T(x) can be viewed as C(E(x)). x -> E -> C -> y.
        The function of this method is to freeze E. It is used by TL defense method
        """
        raise NotImplementedError()