from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module, parallel


class BaseHook(metaclass=ABCMeta):
    """Monitor the model when forward"""

    def __init__(self, module: Module) -> None:
        self.hook = module.register_forward_hook(self._hook_gather_impl)
        self.features = None

    def _hook_gather_impl(self, module, input, output):
        feature = self.hook_fn(module, input, output)
        self.features = feature

    @abstractmethod
    def hook_fn(self, module, input, output):
        raise NotImplementedError()

    def get_feature(self) -> Tensor:
        """
        Returns:
            Tensor: the value that the hook monitor.
        """
        return self.features

    def close(self):
        self.hook.remove()


class OutputHook(BaseHook):
    """Monitor the output of the model"""

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    def hook_fn(self, module, input, output):
        return output


class InputHook(BaseHook):
    """Monitor the input of the model"""

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    def hook_fn(self, module, input, output):
        return input


class FirstInputHook(BaseHook):
    """Monitor the input of the model"""

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    def hook_fn(self, module, input, output):
        return input[0]


class DeepInversionBNFeatureHook(BaseHook):
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        super().__init__(module)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2
        )

        return r_feature
        # must have no output
