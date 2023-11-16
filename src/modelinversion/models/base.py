from torch import nn
from abc import ABCMeta, abstractmethod

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