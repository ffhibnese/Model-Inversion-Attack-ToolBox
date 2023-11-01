from dataclasses import dataclass
import torch

@dataclass
class ModelResult:
    result: torch.Tensor
    feat: list
    addition_info: dict = None