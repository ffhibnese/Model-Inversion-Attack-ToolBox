import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from modelinversion.datasets import FaceScrub224
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from kornia import augmentation

from modelinversion.models import (
    LoktGenerator256,
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
    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'

    target_model_ckpt_path = f'/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    experiment_dir = f'./results_attack/lokt_{tag}'

    # if tag == 'no':
    #     tag = ''
    # else:
    tag = f'_{tag}'
    num_classes = 1000
    generator_ckpt_path = f'./gan/lokt_ffhq256_facescrub224_ir152{tag}_gan/G.pth'
    aug_model_names = ['densenet121', 'densenet161', 'densenet169']
    aug_model_ckpt_paths = [
        f'./classifier/lokt_ffhq256_facescrub224_ir152{tag}/densenet121/facescrub224_densenet121.pth',
        f'./classifier/lokt_ffhq256_facescrub224_ir152{tag}/densenet161/facescrub224_densenet161.pth',
        f'./classifier/lokt_ffhq256_facescrub224_ir152{tag}/densenet169/facescrub224_densenet169.pth',
    ]
    # target_model_ckpt_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth'
    eval_model_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_maxvit_t/facescrub224_maxvit_t.pth'
    eval_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    attack_targets = list(range(100))

    batch_size = 16

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
    # generator = LoktGenerator256(num_classes)
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

    eval_dataset = FaceScrub224(
        eval_dataset_path,
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
    # 'vib0.01',
    # 'bido0.01_0.1_pretrain',
    # 'ls0.05',
    # 'tl0.5',
    # 'rolss0.0_2',
]:

    # for tag in tags:
    main(tag, 7)
