import sys
import os
import time

sys.path.append('../../../src')

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
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

from modelinversion.models import IR152_64, LoraWrapper, auto_classifier_from_pretrained
from modelinversion.train import SimpleTrainer, SimpleTrainConfig
from modelinversion.utils import (
    Logger,
    LabelSmoothingCrossEntropyLoss,
    unfreeze,
    freeze,
    traverse_module,
    Accumulator,
)
from modelinversion.datasets import FaceScrub224


def calculate_entropy(y):
    _, counts = torch.unique(y, return_counts=True)
    probs = counts.float() / len(y)
    return -torch.sum(probs * torch.log(probs))


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@torch.no_grad()
def main(tag, train=False):


    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    dataset_path = '/mnt/data/<usrname>/datasets/facescrub'

    batch_size = 500

    device = 'cuda'
    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    if hasattr(target_model, 'unwrap'):
        target_model = target_model.unwrap()
    target_model: IR152_64 = target_model.to(device)
    from modelinversion.models import NeckWrapper

    # if isinstance(target_model, NeckWrapper):
    #     target_model.module.fc_layer = nn.Identity()
    # else:
    #     target_model.fc_layer = nn.Identity()
    target_model.eval()
    freeze(target_model)

    train_dataset = FaceScrub224(
        dataset_path,
        train=train,
        output_transform=Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # RandomHorizontalFlip(p=0.5),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
    )
    from tqdm import tqdm

    all_confs = []

    for i, (x, y) in enumerate(tqdm(train_loader)):

        target_model.zero_grad()

        x, y = x.to(device), y.to(device)
        x = x.detach()

        pred = target_model(x)
        if not isinstance(pred, torch.Tensor):
            pred = pred[0]
        pred = F.softmax(pred)
        # get the confidence of the prediction
        confidence = torch.max(pred, dim=1).values

        all_confs.append(confidence.cpu())

    all_confs = torch.cat(all_confs, dim=0)

    import numpy as np

    print(torch.mean(all_confs))


# 'no', 'bido', 'vib0.1', 'tl0.4', 'lora2', 'ls0.3',
if __name__ == '__main__':
    # for tag in ['neck20none', 'neck20tanh']:
    #     for train in [True, False]:
    #         main(tag, train)
    # for tag in ['neck50tanh', 'neck50tanh_focal8_lr2e-05_5', 'no']:
    #     main(tag, train=True)
    main('neck50tanh_focal8_lr0.002_5', True)
