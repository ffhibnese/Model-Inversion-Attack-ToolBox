import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from modelinversion.datasets import FaceScrub112
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
    ImageFidPRDCMetric,
)


def main(tag, cuda):

    device_ids_available = f'{cuda}'

    # if tag == 'no':
    #     target_model_ckpt_path = f'../../../checkpoints_v2/classifier/facescrub64/facescrub64_ir152_98.25.pth'
    # elif tag == 'tl0.5':
    #     target_model_ckpt_path = f'/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/facescrub64/facescrub64_ir152_tl_0.5_95.36.pth'
    # else:
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    experiment_dir = f'./results/{tag}'

    # if tag == 'no':
    #     tag = ''
    # else:
    tag = f'_{tag}'
    num_classes = 1000
    generator_ckpt_path = f'./gan/lokt_ffhq64_facescrub64_ir152{tag}_gan/G.pth'
    aug_model_names = ['densenet121', 'densenet161', 'densenet169']
    aug_model_ckpt_paths = [
        f'./classifier/lokt_ffhq64_facescrub64_ir152{tag}/densenet121/facescrub64_densenet121.pth',
        f'./classifier/lokt_ffhq64_facescrub64_ir152{tag}/densenet161/facescrub64_densenet161.pth',
        f'./classifier/lokt_ffhq64_facescrub64_ir152{tag}/densenet169/facescrub64_densenet169.pth',
    ]
    # target_model_ckpt_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    eval_model_ckpt_path = '../../../checkpoints_v2/classifier/facescrub112/facescrub112_facenet112_99.38.pth'
    eval_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    attack_targets = list(range(100))

    batch_size = 168

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

    eval_model.eval()
    generator.eval()

    # prepare eval dataset

    eval_dataset = FaceScrub112(
        eval_dataset_path,
        train=True,
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

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=10,
        optimize_batch_size=batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
        save_optimized_images=True,
        save_final_images=False,
        eval_metrics=[
            target_accuracy_metric,
            accuracy_metric,
            distance_metric,
            fid_prdc_metric,
        ],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)

    logger.close()


for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:

    # for tag in tags:
    main(tag, 7)
