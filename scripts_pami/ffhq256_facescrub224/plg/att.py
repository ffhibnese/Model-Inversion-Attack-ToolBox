import sys
import os
import argparse
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from modelinversion.datasets import FaceScrub224
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from kornia import augmentation

from modelinversion.models import (
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import (
    unwrapped_parallel_module,
    augment_images_fn_generator,
    Logger,
    freeze,
)
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)


def main(tag):
    paths = get_attack_paths('plg', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device
    num_classes = 530
    attack_targets = list(range(100))
    batch_size = 16
    num_classes = 530

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 128

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path, register_last_feature_hook=True
    )
    # eval_model_2 = auto_classifier_from_pretrained(
    #     eval_model_ckpt_path_2, register_last_feature_hook=True
    # )
    generator = auto_generator_from_pretrained(paths.generator_ckpt_path)

    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    # eval_model_2 = nn.parallel.DataParallel(eval_model_2, device_ids=gpu_devices).to(
    #     device
    # )
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    target_model.eval()
    eval_model.eval()
    # eval_model_2.eval()
    generator.eval()

    freeze(generator)
    freeze(target_model)
    freeze(eval_model)
    # freeze(eval_model_2)

    # prepare eval dataset

    eval_dataset = FaceScrub224(
        paths.eval_dataset_path,
        train=True,
        output_transform=Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    # prepare optimization

    create_aug_images_fn = augment_images_fn_generator(
        None,
        add_origin_image=False,
        augment=augmentation.container.ImageSequential(
            augmentation.RandomResizedCrop(
                (224, 224), scale=(0.8, 1.0), ratio=(1.0, 1.0)
            ),
            augmentation.ColorJitter(brightness=0.2, contrast=0.2),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(5),
        ),
        augment_times=2,
    )

    optimization_config = ImageAugmentWhiteBoxOptimizationConfig(
        experiment_dir=paths.experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.1},
        loss_fn='max_margin',
        create_aug_images_fn=create_aug_images_fn,
        iter_times=600,
    )

    optimization_fn = ImageAugmentWhiteBoxOptimization(
        optimization_config, generator, target_model
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

    # accuracy_metric_2 = ImageClassifierAttackAccuracy(
    #     batch_size, eval_model_2, device=device, description='evaluation-incv3'
    # )

    # distance_metric_2 = ImageDistanceMetric(
    #     batch_size,
    #     eval_model_2,
    #     eval_dataset,
    #     device=device,
    #     description='evaluation-incv3',
    # )

    fid_prdc_metric = ImageFidPRDCMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=paths.experiment_dir,
        fid=True,
        prdc=True,
    )

    # prepare attack

    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=paths.experiment_dir,
    )

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
            # accuracy_metric_2,
            # distance_metric_2,
        ],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)


for tag in ALL_TAGS:
    main(tag)
