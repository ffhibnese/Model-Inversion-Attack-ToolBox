import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torchvision.transforms import (
    ToTensor,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    CenterCrop,
    Resize,
    functional as TF,
)

from modelinversion.models import (
    get_stylegan2ada_generator,
    auto_classifier_from_pretrained,
    TorchvisionClassifierModel,
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
    freeze,
    traverse_module,
)
from modelinversion.attack import (
    IntermediateWhiteboxOptimizationConfig,
    StyelGANIntermediateWhiteboxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ImageAugmentClassificationLoss,
)
from modelinversion.datasets import FaceScrub224, ClassSubset
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)


@torch.no_grad()
def main(tag, feature_compressed=False):

    save_name = f'{tag}.pth'
    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'

    experiment_dir = f'./results_weight'
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)

    all_weights = []

    def travel_fn(module):
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.cpu().numpy())

    traverse_module(model, travel_fn)

    os.makedirs(experiment_dir, exist_ok=True)

    torch.save(
        all_weights[-1],
        os.path.join(experiment_dir, save_name),
    )


if __name__ == '__main__':
    device_ids_available = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available

    for tag in ['no', 'neck50tanh', 'bido_ih0.15_oh1.5', 'ls0.01', 'tl0.7', 'vib0.005']:
        main(tag)
