
from torch import nn, Module, Tensor
from torch.optim import Optimizer

def train_mapping_model(
    mapping_module: Module,
    optimizer: Optimizer,
    src_model: Module,
    dst_model: Module,
):
    pass