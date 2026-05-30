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
from modelinversion.utils import augment_images_fn_generator, Logger, freeze
from modelinversion.attack import (
    IntermediateWhiteboxOptimizationConfig,
    StyelGANIntermediateWhiteboxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ImageAugmentClassificationLoss,
)
from modelinversion.datasets import CelebA224, ClassSubset
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)


def main(tag):

    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'

    device_ids_available = '0'

    experiment_dir = f'./if_resnet152{tag}'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/ffhq.pkl'
    )
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    # '/mnt/data/<usrname>/Model-Inversion-Attack-ToolBox/results/train_facescrub64_ir152_lora/facescrub64_ir152_lora.pth'
    eval_model_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/facescrub224/facescrub224_inception_v3_94.45.pth'
    eval_model_ckpt_path_2 = '/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_maxvit_t/facescrub224_maxvit_t.pth'
    eval_dataset_path = '/mnt/data/<usrname>/datasets/pre_celeba_high/private_train'
    attack_targets = list(range(100))

    sample_batch_size = 16
    optimize_batch_size = 6
    evaluation_batch_size = 4
    sample_num = 5000
    optimize_num = 5

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    target_resolution = 224
    eval_resolution = 224

    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset

    eval_dataset = CelebA224(
        eval_dataset_path,
        # train=True,
        output_transform=Compose(
            [
                Resize((eval_resolution, eval_resolution)),
                # ToTensor(),
                # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    for label in list(range(30, 60)):
        label_dataset = ClassSubset(eval_dataset, [label])

        os.makedirs(f'./celeba60/{label}', exist_ok=True)
        for i, (img, _) in enumerate(label_dataset):
            img.save(f'./celeba60/{label}/{i}.png')

    logger.close()


if __name__ == '__main__':
    tags = [
        # 'ls0.01'
        # 'tl0.5_focal3_freemix0.5'
        # 'tl0.5_focal3_ramdomfreemix0.5_cenls0',
        # 'tl0.5_focal3_ramdomfixmix0.5_cenls0'
        # 'ls0.005',
        # 'bido_ih0.05_oh0.5'
        # 'tl0.5_focal3_ramdomfixmix0.5_cenls0.05'
        # 'neck100tanh'
        # 'neck50tanh'
        # 'lora10_neck50tanh'
        # 'neck40tanh'
        # 'lora10end0.9_neck40tanh'
        # 'neck50tanh_focal8_lr0.002_5'
        # 'neck50tanh_focal8_lr2e-05_5',
        # 'tl0.7'
        # 'no'
        # 'tl0.4',
        # 'vib0.005'
        # 'bido_ih0.15_oh1.5'
        # 'ftneck35tanh',
        # 'ftneck40tanh'
        # 'ftneck40tanh_focal8_lr2e-05_5'
        'neck50none',
        # 'neck50leaky_relu',
        # 'neck50relu',
        # 'neck50relu6',
        # 'neck50sigmoid',
    ]
    for tag in tags:
        main(tag)
