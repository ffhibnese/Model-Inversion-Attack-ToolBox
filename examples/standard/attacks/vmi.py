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
)
from modelinversion.sampler import (
    LayeredFlowLatentsSampler,
    GaussianMixtureLatentsSampler,
)
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
)
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.datasets import CelebA112
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
)

if __name__ == '__main__':

    device_ids_available = '5'
    num_classes = 1000

    experiment_dir = '/data/qyx/Model-Inversion-Attack-ToolBox/test/vmi'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/stylegan2_ada'
    stylegan2ada_ckpt_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/neurips2021-celeba-stylegan/network-snapshot-002298.pkl'
    target_model_name = 'resnet34'
    target_model_ckpt_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/neurips2021-celeba-cls/best_ckpt.pt'
    eval_model_name = 'ir_se'
    eval_model_ckpt_path = '<fill it>'
    eval_dataset_path = '<fill it>'
    attack_targets = list(range(100))

    # prepare flow params
    sample_batch_size = 16
    permute = 'shuffle'
    K = 10
    glow = True
    coupling = 'additive'
    L = 3
    use_actnorm = True
    l_identity = '0-9'

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    mapping, generator = get_stylegan2ada_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, single_w=True
    )

    target_resolution = 64
    eval_resolution = 112

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
    )

    mapping = nn.parallel.DataParallel(mapping, device_ids=gpu_devices).to(device)
    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    mapping.eval()
    target_model.eval()
    eval_model.eval()
    generator.eval()

    # prepare eval dataset

    eval_dataset = CelebA112(
        eval_dataset_path,
        output_transform=ToTensor(),
    )

    # prepare latent sampler

    w_dim = mapping.module.w_dim
    z_dim = mapping.module.z_dim
    num_ws = mapping.module.num_ws

    latent_sampler = LayeredFlowLatentsSampler(
        input_size=w_dim,
        batch_size=sample_batch_size,
        generator=generator,
        k=z_dim,
        l=num_ws,
        flow_permutation=permute,
        flow_K=K,
        flow_glow=glow,
        flow_coupling=coupling,
        flow_L=L,
        flow_use_actnorm=use_actnorm,
        l_identity=l_identity,
        device=device,
        latents_mapping=mapping
    )
    
    print(latent_sampler(attack_targets, sample_num=64))
