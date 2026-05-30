import sys
import sys
import os
import argparse
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize

from modelinversion.models import (
    SimpleGenerator64,
    GmiDiscriminator64,
    IR152_64,
    FaceNet112,
    EfficientNet_b0_64,
    EfficientNet_b1_64,
    EfficientNet_b2_64,
    generate_feature_statics,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
    auto_discriminator_from_pretrained,
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import Logger, freeze
from modelinversion.attack import (
    SimpleWhiteBoxOptimization,
    SimpleWhiteBoxOptimizationConfig,
    GmiDiscriminatorLoss,
    ImageAugmentClassificationLoss,
    ClassificationWithFeatureDistributionLoss,
    ComposeImageLoss,
    ComposeImageLoss,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)
from modelinversion.datasets import FaceScrub224


def main(tag):
    paths = get_attack_paths('lomma_lgmi', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device
    num_classes = 530
    attack_targets = list(range(100))
    batch_size = 10

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 100

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    # target_model =  IR152_64(num_classes=num_classes, register_last_feature_hook=True)
    target_model = auto_classifier_from_pretrained(
        paths.target_model_ckpt_path, register_last_feature_hook=True
    )
    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path, register_last_feature_hook=True
    )
    generator = auto_generator_from_pretrained(paths.generator_ckpt_path)
    discriminator = auto_discriminator_from_pretrained(paths.discriminator_ckpt_path)

    aug_model_0 = auto_classifier_from_pretrained(paths.lomma_aug_model_ckpt_paths[0])
    aug_model_1 = auto_classifier_from_pretrained(paths.lomma_aug_model_ckpt_paths[1])
    aug_model_2 = auto_classifier_from_pretrained(paths.lomma_aug_model_ckpt_paths[2])
    freeze(target_model)
    freeze(eval_model)
    freeze(generator)
    freeze(discriminator)
    freeze(aug_model_0)
    freeze(aug_model_1)
    freeze(aug_model_2)

    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    eval_model = nn.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=gpu_devices).to(device)
    aug_model_0 = nn.DataParallel(aug_model_0, device_ids=gpu_devices).to(device)
    aug_model_1 = nn.DataParallel(aug_model_1, device_ids=gpu_devices).to(device)
    aug_model_2 = nn.DataParallel(aug_model_2, device_ids=gpu_devices).to(device)

    target_model.eval()
    eval_model.eval()
    generator.eval()
    discriminator.eval()
    aug_model_0.eval()
    aug_model_1.eval()
    aug_model_2.eval()

    # prepare eval dataset

    eval_dataset = FaceScrub224(
        paths.eval_dataset_path,
        train=True,
        output_transform=ToTensor(),
    )

    # prepare feature statics
    public_dataset = ImageFolder(paths.lomma_public_dataset_path, transform=ToTensor())
    public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True)

    feature_mean, feature_std = generate_feature_statics(
        public_loader, 5000, target_model, device
    )
    feature_mean, feature_std = feature_mean.to(device), feature_std.to(device)

    # prepare optimization

    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=paths.experiment_dir,
        device=device,
        optimizer='SGD',
        optimizer_kwargs={'lr': 0.02, 'momentum': 0.9},
        iter_times=1500,
    )

    loss_fns = []

    main_iden_loss_fn = ClassificationWithFeatureDistributionLoss(
        target_model,
        feature_mean,
        feature_std,
        classification_loss_fn='nll_loss',
        create_aug_images_fn=None,
        feature_loss_weight=0.4,
    )
    loss_fns.append(main_iden_loss_fn)

    for aug_model in [aug_model_0, aug_model_1, aug_model_2]:
        aug_loss_fn = ImageAugmentClassificationLoss(
            classifier=aug_model, loss_fn='nll_loss', create_aug_images_fn=None
        )
        loss_fns.append(aug_loss_fn)

    discriminator_loss_fn = GmiDiscriminatorLoss(discriminator)
    loss_fns.append(discriminator_loss_fn)

    loss_fn = ComposeImageLoss(loss_fns, weights=[25, 25, 25, 25, 1])

    optimization_fn = SimpleWhiteBoxOptimization(
        optimization_config, generator, loss_fn
    )

    # prepare metrics

    to_eval_transform = Resize((112, 112), antialias=True)

    accuracy_metric = ImageClassifierAttackAccuracy(
        batch_size,
        eval_model,
        device=device,
        description='evaluation',
        transform=to_eval_transform,
    )

    distance_metric = ImageDistanceMetric(
        batch_size,
        eval_model,
        eval_dataset,
        device=device,
        description='evaluation',
        transform=to_eval_transform,
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
    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=paths.experiment_dir,
    )

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=5,
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
    # attacker.evaluate_from_pre_generate(generator, list(range(530)), 'alt530', device)
    # attacker.evaluate_from_pre_generate(generator, list(range(100)), 'alt100', device)


for tag in ALL_TAGS:
    main(tag)
