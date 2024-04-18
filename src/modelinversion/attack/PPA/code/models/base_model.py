from abc import abstractmethod

import numpy as np
import torch


class BaseModel(torch.nn.Module):
    """
    Base model for all PyTorch models.
    """
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            'cuda:0') if self.use_cuda else torch.device('cpu')

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def set_parameter_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(param.numel() for param in self.parameters()
                       if param.requires_grad)
        return sum(param.numel() for param in self.parameters())

    def __str__(self):
        num_params = np.sum([param.numel() for param in self.parameters()])
        if self.name:
            return self.name + '\n' + super().__str__(
            ) + f'\n Total number of parameters: {num_params}'
        else:
            return super().__str__(
            ) + f'\n Total number of parameters: {num_params}'
