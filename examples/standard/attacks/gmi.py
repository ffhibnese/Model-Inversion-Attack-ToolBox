import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from modelinversion.datasets import CelebA112
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
    auto_discriminator_from_pretrained,
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import Logger
from modelinversion.attack import (
    SimpleWhiteBoxOptimization,
    SimpleWhiteBoxOptimizationConfig,
    GmiDiscriminatorLoss,
    ImageAugmentClassificationLoss,
    ComposeImageLoss,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
)


if __name__ == '__main__':

    # prepare path args

    experiment_dir = '<fill it>'
    device_ids_available = '0'
    num_classes = 1000
    generator_ckpt_path = '<fill it>'
    discriminator_ckpt_path = '<fill it>'
    target_model_ckpt_path = '<fill it>'
    eval_model_ckpt_path = '<fill it>'
    eval_dataset_path = '<fill it>'
    attack_targets = list(range(1000))

    batch_size = 200

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 100

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
    )
    generator = auto_generator_from_pretrained(generator_ckpt_path)
    discriminator = auto_discriminator_from_pretrained(discriminator_ckpt_path)

    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    eval_model = nn.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)

    target_model.eval()
    eval_model.eval()
    generator.eval()
    discriminator.eval()

    # prepare eval dataset

    eval_dataset = CelebA112(
        eval_dataset_path,
        output_transform=ToTensor(),
    )

    # prepare optimization

    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='SGD',
        optimizer_kwargs={'lr': 0.02, 'momentum': 0.9},
        iter_times=1500,
    )

    identity_loss_fn = ImageAugmentClassificationLoss(
        classifier=target_model, loss_fn='ce', create_aug_images_fn=None
    )

    discriminator_loss_fn = GmiDiscriminatorLoss(discriminator)

    loss_fn = ComposeImageLoss(
        [identity_loss_fn, discriminator_loss_fn], weights=[100, 1]
    )

    optimization_fn = SimpleWhiteBoxOptimization(
        optimization_config, generator, loss_fn
    )

    # prepare metrics

    accuracy_metric = ImageClassifierAttackAccuracy(
        batch_size, eval_model, device=device, description='evaluation'
    )

    distance_metric = ImageDistanceMetric(
        batch_size,
        eval_model,
        eval_dataset,
        device=device,
        description='evaluation',
        save_individual_res_dir=experiment_dir,
    )

    fid_prdc_metric = ImageFidPRDCMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=experiment_dir,
        fid=True,
        prdc=True,
    )

    # prepare attack

    attack_config = ImageClassifierAttackConfig(
        # attack args
        latents_sampler,
        optimize_num=50,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        # save path args
        save_dir=experiment_dir,
        save_optimized_images=True,
        save_final_images=False,
        # metric args
        eval_metrics=[accuracy_metric, distance_metric, fid_prdc_metric],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
