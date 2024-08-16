import importlib
from typing import Callable

import torch
from torch import nn
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
    v = torch.clip(torch.eye(outputs.shape[-1])[targets.detach().cpu()] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) * (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()


_LOSS_MAPPING = {
    'ce': F.cross_entropy,
    'poincare': poincare_loss,
    'max_margin': max_margin_loss,
}


class LabelSmoothingCrossEntropyLoss:
    """The Cross Entropy Loss with label smoothing technique. Used in the LS defense method."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = label_smoothing

    def __call__(self, inputs, labels, reduction='mean'):
        ls = self.label_smoothing
        confidence = 1.0 - ls
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + ls * smooth_loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            # assert loss.ndim == 2, f'Invalid loss shape: {loss.shape}'
            return loss
        raise RuntimeError(f'Invalid reduction mode: {reduction}')

# class SoftTargetCrossEntropy(nn.Module):


class InverseFocalLoss:

    def __init__(self, gamma: float = 4, alpha: float = 0.25, label_smoothing=0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        # self.ce_fn = LabelSmoothingCrossEntropyLoss(label_smoothing=label_smoothing)

    def __call__(self, inputs, targets):

        ls = self.label_smoothing
        confidence = 1.0 - ls
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        
        # ce_loss = self.ce_fn(inputs, targets, reduction='none')
        pt = torch.exp(-nll_loss)
        focal_loss = -self.alpha * pt**self.gamma * nll_loss

        loss = confidence * focal_loss + ls * smooth_loss

        return torch.mean(loss)


class TorchLoss:
    """Find loss function from 'torch.nn.functional' and 'torch.nn'"""

    def __init__(self, loss_fn: str | Callable, *args, **kwargs) -> None:
        # super().__init__()
        self.fn = None
        if isinstance(loss_fn, str):
            if loss_fn.lower() in _LOSS_MAPPING:
                self.fn = _LOSS_MAPPING[loss_fn.lower()]
            else:
                module = importlib.import_module('torch.nn.functional')
                fn = getattr(module, loss_fn, None)
                if fn is not None:
                    self.fn = lambda *arg, **kwd: fn(*arg, *args, **kwd, **kwargs)
                else:
                    module = importlib.import_module('torch.nn')
                    t = getattr(module, loss_fn, None)
                    if t is not None:
                        self.fn = t(*args, **kwargs)
                if self.fn is None:
                    raise RuntimeError(f'loss_fn {loss_fn} not found.')
        else:
            self.fn = loss_fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
