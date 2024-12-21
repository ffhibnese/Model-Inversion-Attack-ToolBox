import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from modelinversion.datasets import CelebA112
from torchvision.transforms import ToTensor, Resize, Compose
from kornia import augmentation

from modelinversion.models import (
    LoktGenerator64,
    TorchvisionClassifierModel,
    FaceNet112,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import (
    unwrapped_parallel_module,
    augment_images_fn_generator,
    Logger,
)
from modelinversion.attack import (
    SimpleWhiteBoxOptimization,
    SimpleWhiteBoxOptimizationConfig,
    ComposeImageLoss,
    ImageAugmentClassificationLoss,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)


if __name__ == '__main__':

    device_ids_available = '2'

    experiment_dir = '<fill it>'
    num_classes = 1000
    generator_ckpt_path = (
        '../../../checkpoints_v2/attacks/lokt/lokt_ffhq64_celeba64_ir152_G.pth'
    )
    aug_model_names = ['densenet121', 'densenet161', 'densenet169']
    aug_model_ckpt_paths = [
        '../../../checkpoints_v2/attacks/lokt/lokt_celeba64_densenet121_celeba64_ir152.pth',
        '../../../checkpoints_v2/attacks/lokt/lokt_celeba64_densenet161_celeba64_ir152.pth',
        '../../../checkpoints_v2/attacks/lokt/lokt_celeba64_densenet169_celeba64_ir152.pth',
    ]
    target_model_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    )
    eval_model_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba112/celeba112_facenet112_97.72.pth'
    )
    eval_dataset_path = '../../../dataset/celeba_low/private_train'
    attack_targets = list(range(1000))

    batch_size = 100

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 128

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    aug_models = []
    for arch_name, ckpt_path in zip(aug_model_names, aug_model_ckpt_paths):
        # model = TorchvisionClassifierModel(
        #     arch_name, num_classes=num_classes, resolution=64
        # )
        # model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
        model = auto_classifier_from_pretrained(ckpt_path)
        model = nn.parallel.DataParallel(model, device_ids=gpu_devices).to(device)
        model.eval()
        aug_models.append(model)

    # eval_model = FaceNet112(num_classes=num_classes, register_last_feature_hook=True)
    # generator = LoktGenerator64(num_classes)
    target_model = auto_classifier_from_pretrained(
        target_model_ckpt_path, register_last_feature_hook=True
    )
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
    )

    generator = auto_generator_from_pretrained(generator_ckpt_path)

    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    target_model.eval()
    eval_model.eval()
    generator.eval()

    # prepare eval dataset

    eval_dataset = CelebA112(
        eval_dataset_path,
        output_transform=ToTensor(),
    )

    # prepare optimization

    create_aug_images_fn = augment_images_fn_generator(
        None,
        add_origin_image=False,
        augment=augmentation.container.ImageSequential(
            augmentation.RandomResizedCrop(
                (64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)
            ),
            augmentation.ColorJitter(brightness=0.2, contrast=0.2),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(5),
        ),
        augment_times=2,
    )

    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.1},
    )

    loss_fns = []
    for aug_model in aug_models:
        single_loss_fn = ImageAugmentClassificationLoss(
            aug_model, 'max_margin', create_aug_images_fn=create_aug_images_fn
        )
        loss_fns.append(single_loss_fn)

    loss_fns_compose = ComposeImageLoss(loss_fns)

    optimization_fn = SimpleWhiteBoxOptimization(
        optimization_config, generator, loss_fns_compose
    )

    # prepare metrics

    target_accuracy_metric = ImageClassifierAttackAccuracy(
        batch_size, target_model, device=device, description='target'
    )

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
    evalraw_path = '/data/<username>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/celeba112/celeba112_facenet112_95.88.tar'
    evalraw_model = auto_classifier_from_pretrained(evalraw_path)
    evalraw_model = nn.DataParallel(evalraw_model, device_ids=gpu_devices).to(device)
    evalraw_model.eval()

    accuracy_raw_metric = ImageClassifierAttackAccuracy(
        batch_size, evalraw_model, device=device, description='evaluation_origin'
    )

    distance_metric_raw = ImageDistanceMetric(
        batch_size,
        evalraw_model,
        eval_dataset,
        device=device,
        description='evaluation_origin',
        save_individual_res_dir=experiment_dir,
    )

    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=experiment_dir,
    )

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=5,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
        save_optimized_images=True,
        save_final_images=False,
        eval_metrics=[
            target_accuracy_metric,
            accuracy_metric,
            accuracy_raw_metric,
            distance_metric,
            distance_metric_raw,
            face_dist_metric,
            fid_prdc_metric,
        ],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
