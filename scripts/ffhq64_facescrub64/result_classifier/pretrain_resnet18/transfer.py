import sys
import os
import time

sys.path.append('/mnt/data/<usrname>/mywork/lora_defense/src')
# import clip
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    Resize,
)

from modelinversion.models import (
    TorchvisionClassifierModel,
    auto_classifier_from_pretrained,
)
from torchvision.models.resnet import ResNet18_Weights, resnet18
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss
from modelinversion.datasets import LabelImageFolder

model: TorchvisionClassifierModel = auto_classifier_from_pretrained(
    "./pretrain_resnet18.pth"
)
model = model.model

state_dict = model.state_dict()
state_dict_keys = list(model.state_dict().keys())

new_state_dict = {}
for key in state_dict_keys:
    if "fc" in key:
        continue
    new_state_dict[key] = state_dict[key]

new_model = resnet18()
load_result = new_model.load_state_dict(new_state_dict, strict=False)

new_model.eval()

print(load_result)

torch.save(new_model.state_dict(), "resnet18_pretrain_original_pt.pt")
