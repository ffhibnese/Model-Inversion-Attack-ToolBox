import importlib

import torch
from torch.nn import functional as F

def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()

_LOSS_MAPPING = {
    'ce': F.cross_entropy,
    'poincare': poincare_loss,
    'max_margin': max_margin_loss
}

class TorchLoss:
    
    def __init__(self, loss_fn: str, *args, **kwargs) -> None:
        # super().__init__()
        
        if isinstance(loss_fn, str):
            if isinstance(loss_fn, str):
                if loss_fn.lower() in _LOSS_MAPPING:
                    self.fn = _LOSS_MAPPING[loss_fn.lower()]
                else:
                    module = importlib.import_module('torch.nn.functional')
                    self.fn = getattr(module, loss_fn)
        else:
            self.fn = loss_fn
            
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)