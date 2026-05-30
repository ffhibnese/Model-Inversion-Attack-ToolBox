import sys
import os
import argparse
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

from modelinversion.models import (
    SimpleGenerator64,
    IR152_64,
    FaceNet112,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
    auto_discriminator_from_pretrained,
)
from modelinversion.sampler import LabelOnlySelectLatentsSampler
from modelinversion.utils import Logger, freeze
from modelinversion.attack import (
    BrepOptimizationConfig,
    BrepOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.scores import ImageClassificationAugmentLabelOnlyScore
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)
from modelinversion.datasets import FaceScrub224


def main(tag):
    paths = get_attack_paths('brep', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device
    num_classes = 530
    attack_targets = list(range(100))
    batch_size = 50

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 100

    # target_model = IR152_64(num_classes=num_classes)
    # eval_model = FaceNet112(num_classes, register_last_feature_hook=True)
    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path, register_last_feature_hook=True
    )
    generator = auto_generator_from_pretrained(paths.generator_ckpt_path)
    discriminator = auto_discriminator_from_pretrained(paths.discriminator_ckpt_path)

    freeze(target_model)
    freeze(eval_model)
    freeze(generator)
    freeze(discriminator)

    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    eval_model = nn.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)
    discriminator.eval()

    target_model.eval()
    eval_model.eval()
    generator.eval()

    latents_sampler = LabelOnlySelectLatentsSampler(
        z_dim, batch_size, generator, target_model, device=device
    )

    # prepare eval dataset

    eval_dataset = FaceScrub224(
        paths.eval_dataset_path,
        train=True,
        output_transform=Compose(
            [Resize((224, 224)), ToTensor(), Normalize((0.5,), (0.5,))]
        ),
    )

    # prepare optimization

    optimization_config = BrepOptimizationConfig(
        experiment_dir=paths.experiment_dir, device=device, iter_times=1000
    )

    image_score_fn = ImageClassificationAugmentLabelOnlyScore(
        classifier=target_model, device=device, correct_score=1, wrong_score=-1
    )

    optimization_fn = BrepOptimization(
        config=optimization_config, generator=generator, image_score_fn=image_score_fn
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
        save_individual_res_dir=paths.experiment_dir,
    )
    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=paths.experiment_dir,
    )

    fid_prdc_metric = ImageFidPRDCMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=paths.experiment_dir,
        fid=True,
        prdc=True,
    )

    # prepare attack

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=10,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        save_dir=paths.experiment_dir,
        save_optimized_images=True,
        save_final_images=False,
        eval_metrics=[
            accuracy_metric,
            distance_metric,
            face_dist_metric,
            fid_prdc_metric,
        ],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
    logger.close()


for tag in ALL_TAGS:
    main(tag)
