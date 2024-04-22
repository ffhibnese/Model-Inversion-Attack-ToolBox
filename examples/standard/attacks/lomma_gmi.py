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
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import (
    Logger,
)
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
)


if __name__ == '__main__':

    experiment_dir = '../../../results/lommagmi_celeba_celeba_ir152'
    device_ids_str = '3'
    num_classes = 1000
    generator_ckpt_path = '/data/yuhongyao/Model-Inversion-Attack-ToolBox/test/history_ckpts/checkpoints/gmi/celeba64_G.pth'  # '/data/yuhongyao/papar_codes/PLG-MI-Attack/plgsmalltest/gen_latest.pth.tar'
    discriminator_ckpt_path = '/data/yuhongyao/Model-Inversion-Attack-ToolBox/test/history_ckpts/checkpoints/gmi/celeba64_D.pth'
    target_model_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    )
    eval_model_ckpt_path = (
        '../../../checkpoints_v2/classifier/celeba112/celeba112_facenet112_97.72.pth'
    )
    aug_model_efficientnet_b0_path = '../../..//checkpoints_v2/attacks/lomma/celeba64_efficientnet_b0_celeba64_ir152.pth'
    aug_model_efficientnet_b1_path = '../../..//checkpoints_v2/attacks/lomma/celeba64_efficientnet_b1_celeba64_ir152.pth'
    aug_model_efficientnet_b2_path = '../../..//checkpoints_v2/attacks/lomma/celeba64_efficientnet_b2_celeba64_ir152.pth'
    public_dataset_path = (
        '/data/yuhongyao/Model-Inversion-Attack-ToolBox/dataset/celeba/split/public'
    )
    eval_dataset_path = '/data/yuhongyao/Model-Inversion-Attack-ToolBox/dataset/celeba/split/private/train'
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

    target_model = IR152_64(num_classes=num_classes, register_last_feature_hook=True)
    eval_model = FaceNet112(num_classes, register_last_feature_hook=True)
    generator = SimpleGenerator64(in_dim=z_dim)
    discriminator = GmiDiscriminator64()

    aug_model_0 = EfficientNet_b0_64(num_classes)
    aug_model_1 = EfficientNet_b1_64(num_classes)
    aug_model_2 = EfficientNet_b2_64(num_classes)

    target_model.load_state_dict(
        torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    )
    eval_model.load_state_dict(
        torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict']
    )
    generator.load_state_dict(
        torch.load(generator_ckpt_path, map_location='cpu')['state_dict']
    )
    discriminator.load_state_dict(
        torch.load(discriminator_ckpt_path, map_location='cpu')['state_dict']
    )
    aug_model_0.load_state_dict(
        torch.load(aug_model_efficientnet_b0_path, map_location='cpu')['state_dict']
    )
    aug_model_1.load_state_dict(
        torch.load(aug_model_efficientnet_b1_path, map_location='cpu')['state_dict']
    )
    aug_model_2.load_state_dict(
        torch.load(aug_model_efficientnet_b2_path, map_location='cpu')['state_dict']
    )

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

    eval_dataset = ImageFolder(eval_dataset_path, transform=ToTensor())

    # prepare feature statics
    public_dataset = ImageFolder(public_dataset_path, transform=ToTensor())
    public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True)

    feature_mean, feature_std = generate_feature_statics(
        public_loader, 5000, target_model, device
    )
    feature_mean, feature_std = feature_mean.to(device), feature_std.to(device)

    # prepare optimization

    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
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
            classifier=target_model, loss_fn='nll_loss', create_aug_images_fn=None
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
        latents_sampler,
        optimize_num=50,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
        save_optimized_images=True,
        save_final_images=False,
        eval_metrics=[accuracy_metric, distance_metric, fid_prdc_metric],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
