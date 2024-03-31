from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch import Tensor

def copy_or_set_(dest, source):
    """
    A workaround to respect strides of :code:`dest` when copying :code:`source`
    (https://github.com/geoopt/geoopt/issues/70)
    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor
    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


class BaseConstraint:
    
    def __init__(self) -> None:
        self.center_tensor = None
        
    # @property
    # def center(self):
    #     return self.center_tensor
    
    def register_center(self, tensor: Tensor):
        self.center_tensor = tensor
    
    def __call__(self, tensor: Tensor, *args: Any, **kwds: Any) -> Any:
        return tensor
    
class MinMaxConstraint(BaseConstraint):
    
    def __init__(self, min_tensor, max_tensor) -> None:
        super().__init__()
        
        self.min_tensor = min_tensor
        self.max_tensor = max_tensor
        
    def register_center(self, tensor: Tensor):
        pass
        
    def __call__(self, tensor: Tensor, *args: Any, **kwds: Any) -> Any:
        res = torch.min(tensor, self.max_tensor)
        res = torch.max(tensor, self.min_tensor)
        return copy_or_set_(tensor, res)
    
class L1ballConstraint(BaseConstraint):
    
    def __init__(self, bias: float) -> None:
        super().__init__()
        
        self.bias = bias
        
    def __call__(self, tensor: Tensor, *args: Any, **kwds: Any) -> Any:
        x = tensor.detach()
        eps = self.bias
        original_shape = x.shape
        x = x.view(x.shape[0], -1)
        mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * torch.sign(x)
        return copy_or_set_(tensor, x.view(original_shape))