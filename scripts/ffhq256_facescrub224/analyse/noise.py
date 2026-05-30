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
    auto_classifier_from_pretrained, TorchvisionClassifierModel
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,freeze
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

from torchvision.utils import save_image
from torchvision.transforms.functional import resize

# @torch.no_grad()
def main(tag, feature_compressed=False):

    save_name = f'{tag}.pth'
    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'


    experiment_dir = f'./results_tsne'
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    model = auto_classifier_from_pretrained(
        target_model_ckpt_path
    ).to(device)
    model.eval()

    # freeze(model)


    noise = torch.randn(1, 3, 224, 224)
    # noise.requires_grad_(True)
    noise = noise.to(device).detach().requires_grad_(True)

    optimizer = torch.optim.SGD([noise], lr=10)
    label = torch.LongTensor([99]).to(device)
    for i in range(1000):
        optimizer.zero_grad()
        pred = model(noise)[0]
        loss = nn.functional.cross_entropy(pred, label)
        conf = torch.exp(-loss)
        loss.backward()

        # print('>>',noise.grad)
        # print()
        
        optimizer.step()
        print(f'iter {i} loss {loss.item()} conf {conf.item()}')
    
    noise = noise.detach().cpu()
    img = resize(noise, (128, 128))
    save_image(img, f'./noise_img99.png')
    save_image(noise, f'./noise99.png')


        

if __name__ == '__main__':
    device_ids_available = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available

    # for tag in ['no', 'neck50tanh','bido_ih0.15_oh1.5','ls0.01','tl0.7','vib0.005']:
    for tag in ['no']:
        main(tag)
