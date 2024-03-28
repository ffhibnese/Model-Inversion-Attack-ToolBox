import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from kornia import augmentation

from modelinversion.models import PlgmiGenerator64, IR152_64, FaceNet112
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import unwrapped_parallel_module, augment_images_fn_generator, Logger
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker
)
from modelinversion.metrics import ImageClassifierAttackAccuracy, ImageDistanceMetric, ImageFidPRDCMetric


if __name__ == '__main__':
    
    
    device_ids_str = '0'
    
    experiment_dir = '<fill it>'
    generator_ckpt_path = '<fill it>'
    target_model_ckpt_path = '<fill it>'
    eval_model_ckpt_path = '<fill it>'
    eval_dataset_path = '<fill it>'
    attack_targets = list(range(50))
    
    batch_size = 50
    num_classes = 1000
    
    # prepare logger
    
    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')
    
    # prepare devices
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]
    
    # prepare models
    
    z_dim = 128
    
    latents_sampler = SimpleLatentsSampler(z_dim)
    
    target_model = IR152_64(num_classes=num_classes)
    eval_model = FaceNet112(num_classes=num_classes, register_last_feature_hook=True)
    generator = PlgmiGenerator64(num_classes)
    
    target_model.load_state_dict(torch.load(target_model_ckpt_path, map_location='cpu')['state_dict'])
    eval_model.load_state_dict(torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict'])
    generator.load_state_dict(torch.load(generator_ckpt_path, map_location='cpu')['state_dict'])
    
    target_model = nn.parallel.DistributedDataParallel(target_model, device_ids=gpu_devices).to(device)
    eval_model = nn.parallel.DistributedDataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=gpu_devices).to(device)
    
    target_model.eval()
    eval_model.eval()
    generator.eval()
    
    # prepare eval dataset
    
    eval_dataset = ImageFolder(eval_dataset_path, transform=ToTensor())
    
    # prepare optimization
    
    create_aug_images_fn = augment_images_fn_generator(
        None, add_origin_image=False,
        augment=augmentation.container.ImageSequential(
            augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            augmentation.ColorJitter(brightness=0.2, contrast=0.2),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(5),
        ),
        augment_times=2
    )
    
    optimization_config = ImageAugmentWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.1},
        loss_fn='max_margin',
        create_aug_images_fn= create_aug_images_fn
    )
    
    optimization_fn = ImageAugmentWhiteBoxOptimization(optimization_config, generator, target_model)
    
    # prepare metrics
    
    accuracy_metric = ImageClassifierAttackAccuracy(
        batch_size, eval_model, device=device, description='evaluation'
    )
    
    
    distance_metric = ImageDistanceMetric(
        batch_size, eval_model, eval_dataset, device=device, description='evaluation', save_individual_res_dir=experiment_dir
    )
    
    fid_prdc_metric = ImageFidPRDCMetric(
        batch_size, eval_dataset, device=device, save_individual_prdc_dir=experiment_dir,
        fid=True, prdc=True
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
        eval_optimized_result=False,
        eval_final_result=True
    )
    
    attacker = ImageClassifierAttacker(attack_config)
    
    attacker.attack(attack_targets)
    
    
    