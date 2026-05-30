import sys
import sys
import os
import argparse
import time

sys.path.append('../../../src')

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
    VarienceWhiteboxOptimization,
    VarienceWhiteboxOptimizationConfig,
    KedmiDiscriminatorLoss,
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
from modelinversion.datasets import FaceScrub112


if __name__ == '__main__':

    tag = 'vib0.1'
    experiment_dir = f'./results_attack/lommaked_ir152_{tag}'
    device_ids_str = '3'
    num_classes = 530
    generator_ckpt_path = (
        f'../ked_defense/results_gan/kedmi_ffhq64_facescrub64_ir152_{tag}_gan/G.pth'
    )
    discriminator_ckpt_path = (
        f'../ked_defense/results_gan/kedmi_ffhq64_facescrub64_ir152_{tag}_gan/D.pth'
    )
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_vib0.1/facescrub64_ir152_vib0.1.pth'
    eval_model_ckpt_path = '../../../checkpoints_v2/classifier/facescrub112/facescrub112_facenet112_99.38.pth'
    eval_dataset_path = (
        '/data/<usrname>/intermediate-MIA/intermediate-MIA/data/facescrub'
    )
    aug_model_efficientnet_b0_path = f'./surr/distill_ffhq64_efficientnet_b0_facescrub64_ir152_{tag}/ffhq64_efficientnet_b0_facescrub64_ir152_{tag}.pth'
    aug_model_efficientnet_b1_path = f'./surr/distill_ffhq64_efficientnet_b1_facescrub64_ir152_{tag}/ffhq64_efficientnet_b1_facescrub64_ir152_{tag}.pth'
    aug_model_efficientnet_b2_path = f'./surr/distill_ffhq64_efficientnet_b2_facescrub64_ir152_{tag}/ffhq64_efficientnet_b2_facescrub64_ir152_{tag}.pth'
    public_dataset_path = '../../../dataset/ffhq64'
    attack_targets = list(range(100))

    batch_size = 100

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    z_dim = 100

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    # target_model =  IR152_64(num_classes=num_classes, register_last_feature_hook=True)
    target_model = auto_classifier_from_pretrained(
        target_model_ckpt_path, register_last_feature_hook=True
    )
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
    )
    generator = auto_generator_from_pretrained(generator_ckpt_path)
    discriminator = auto_discriminator_from_pretrained(discriminator_ckpt_path)

    aug_model_0 = auto_classifier_from_pretrained(aug_model_efficientnet_b0_path)
    aug_model_1 = auto_classifier_from_pretrained(aug_model_efficientnet_b1_path)
    aug_model_2 = auto_classifier_from_pretrained(aug_model_efficientnet_b2_path)

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

    eval_dataset = FaceScrub112(
        eval_dataset_path,
        train=True,
        output_transform=ToTensor(),
    )

    # prepare feature statics
    public_dataset = ImageFolder(public_dataset_path, transform=ToTensor())
    public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True)

    feature_mean, feature_std = generate_feature_statics(
        public_loader, 5000, target_model, device
    )
    feature_mean, feature_std = feature_mean.to(device), feature_std.to(device)

    # prepare optimization

    optimization_config = VarienceWhiteboxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.02},
        iter_times=1500,
        generate_num=5,
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

    discriminator_loss_fn = KedmiDiscriminatorLoss(discriminator)
    loss_fns.append(discriminator_loss_fn)

    loss_fn = ComposeImageLoss(loss_fns, weights=[25, 25, 25, 25, 1])

    optimization_fn = VarienceWhiteboxOptimization(
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

    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=experiment_dir,
    )

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=1,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
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
    # attacker.evaluate_from_pre_generate(generator, list(range(530)), 'alt530', device)
    # attacker.evaluate_from_pre_generate(generator, list(range(100)), 'alt100', device)
