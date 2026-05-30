import sys
import os
import argparse
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torch.nn import functional as F
from modelinversion.datasets import FaceScrub224
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
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
    auto_adapter_from_pretrained,
    TorchvisionClassifierModel,
)
from modelinversion.sampler import (
    ImageAugmentSelectLatentsSampler,
    SimpleLatentsSampler,
)
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
    MinMaxConstraint,
    freeze,
)
from modelinversion.attack import (
    C2fGeneticOptimization,
    C2fGeneticOptimizationConfig,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.scores import (
    ImageClassificationAugmentConfidence,
    ImageClassificationAugmentLossScore,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)


def main(tag):
    paths = get_attack_paths('c2f', tag)

    device_ids_available = '5,4,6,0'
    # num_classes = 1000

    attack_targets = list(range(100))

    sample_batch_size = 2
    optimize_batch_size = 2
    final_selection_batch_size = 2
    evaluation_batch_size = 2
    sample_num = 5000

    optimize_num = 32
    # final_num = 50

    w_bound_sample_num = 5000
    p_std_ce = 1

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(paths.experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    latents_mapping, generator = get_stylegan2ada_generator(
        paths.stylegan2ada_path, paths.stylegan2ada_ckpt_path, single_w=True
    )

    target_resolution = 224
    eval_resolution = 224

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path,
        register_last_feature_hook=True,
    )
    embed_model = auto_classifier_from_pretrained(paths.c2f_embed_model_ckpt_path)
    pred_mapping = auto_adapter_from_pretrained(paths.c2f_pred_mapping_ckpt_path)

    # print(torch.load(target_model_ckpt_path, map_location='cpu').keys())

    latents_mapping = nn.parallel.DataParallel(
        latents_mapping, device_ids=gpu_devices
    ).to(device)
    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    pred_mapping = nn.DataParallel(pred_mapping, device_ids=gpu_devices).to(device)
    embed_model = nn.DataParallel(embed_model, device_ids=gpu_devices).to(device)
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    embed_model.eval()
    pred_mapping.eval()
    latents_mapping.eval()
    target_model.eval()
    eval_model.eval()
    generator.eval()

    freeze(embed_model)
    freeze(pred_mapping)
    freeze(latents_mapping)
    freeze(target_model)
    freeze(eval_model)
    freeze(generator)

    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset

    eval_dataset = FaceScrub224(
        paths.eval_dataset_path,
        train=True,
        output_transform=Compose(
            [
                # Resize((eval_resolution, eval_resolution)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    # prepare latent sampler

    w_dim = latents_mapping.module.w_dim

    gan_to_target_transform = Compose(
        [
            CenterCrop((800, 800)),
            Resize((target_resolution, target_resolution), antialias=True),
        ]
    )

    gan_to_embed_transform = Compose(
        [
            CenterCrop((800, 800)),
            Resize((160, 160), antialias=True),
        ]
    )

    latents_sampler = SimpleLatentsSampler(
        w_dim, sample_batch_size, latents_mapping=latents_mapping
    )

    # prepare optimization

    optimize_create_aug_images_fn = augment_images_fn_generator(
        initial_transform=gan_to_target_transform,
        add_origin_image=True,
        # augment=RandomHorizontalFlip(),
        # augment_times=1,
    )

    image_score_fn = ImageClassificationAugmentLossScore(
        model=target_model,
        device=device,
        create_aug_images_fn=optimize_create_aug_images_fn,
        loss_fn='cross_entropy',
    )

    optimization_config = C2fGeneticOptimizationConfig(
        experiment_dir=paths.experiment_dir,
        device=device,
        batch_size=optimize_batch_size,
        final_num=10,
    )

    optimization_fn = C2fGeneticOptimization(
        optimization_config,
        generator,
        image_score_fn=image_score_fn,
        embed_module=embed_model,
        mapping_module=pred_mapping,
        gan_to_embeded_transform=gan_to_embed_transform,
    )

    # prepare metrics

    to_eval_transform = Compose(
        [
            CenterCrop((800, 800)),
            Resize((eval_resolution, eval_resolution), antialias=True),
        ]
    )

    accuracy_metric = ImageClassifierAttackAccuracy(
        evaluation_batch_size,
        eval_model,
        device=device,
        description='evaluation',
        transform=to_eval_transform,
    )

    distance_metric = ImageDistanceMetric(
        evaluation_batch_size,
        eval_model,
        eval_dataset,
        device=device,
        description='evaluation',
        save_individual_res_dir=paths.experiment_dir,
        transform=to_eval_transform,
    )

    fid_prdc_metric = ImageFidPRDCMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=paths.experiment_dir,
        fid=True,
        prdc=True,
        transform=to_eval_transform,
    )

    face_dist_metric = FaceDistanceMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=paths.experiment_dir,
        transform=to_eval_transform,
    )

    # prepare attack

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=optimize_num,
        optimize_batch_size=optimize_num,
        optimize_fn=optimization_fn,
        save_dir=paths.experiment_dir,
        save_optimized_images=True,
        save_final_images=True,
        save_kwargs={'normalize': True},
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
