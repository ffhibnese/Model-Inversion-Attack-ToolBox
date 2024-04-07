import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from kornia import augmentation

from modelinversion.models import SimpleGenerator64, IR152_64, FaceNet112
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import (
    unwrapped_parallel_module,
    augment_images_fn_generator,
    Logger,
    max_margin_loss,
)
from modelinversion.attack import (
    RlbOptimization,
    RlbOptimizationConfig,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ComposeImageLoss,
    ImageAugmentClassificationLoss,
)
from modelinversion.scores import ImageClassificationAugmentLabelOnlyScore
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
)


if __name__ == '__main__':

    experiment_dir = '<fill it>'
    device_ids_str = '1'
    num_classes = 1000
    generator_ckpt_path = '<fill it>'
    target_model_ckpt_path = '<fill it>'
    eval_model_ckpt_path = '<fill it>'
    eval_dataset_path = '<fill it>'
    attack_targets = list(range(10))

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

    target_model = IR152_64(num_classes=num_classes)
    eval_model = FaceNet112(num_classes, register_last_feature_hook=True)
    generator = SimpleGenerator64(in_dim=z_dim)

    target_model.load_state_dict(
        torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    )
    eval_model.load_state_dict(
        torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict']
    )
    generator.load_state_dict(
        torch.load(generator_ckpt_path, map_location='cpu')['state_dict']
    )

    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    eval_model = nn.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.DataParallel(generator, device_ids=gpu_devices).to(device)

    target_model.eval()
    eval_model.eval()
    generator.eval()

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)

    # prepare eval dataset

    eval_dataset = ImageFolder(eval_dataset_path, transform=ToTensor())

    # prepare optimization

    optimization_config = RlbOptimizationConfig(
        experiment_dir=experiment_dir, device=device
    )

    def state_loss(inputs, labels):
        loss1 = F.cross_entropy(inputs, labels)
        loss2 = max_margin_loss(F.softmax(inputs, dim=-1), labels)
        loss2 = 4 * torch.log(torch.clamp_min_(loss2, 1e-7))
        return loss1 + 4 * loss2

    state_loss_fn = ImageAugmentClassificationLoss(
        classifier=target_model, loss_fn=state_loss
    )

    action_loss_fn = ImageAugmentClassificationLoss(
        classifier=target_model, loss_fn='cross_entropy'
    )

    optimization_fn = RlbOptimization(
        config=optimization_config,
        generator=generator,
        state_image_loss_fn=state_loss_fn,
        action_image_loss_fn=action_loss_fn,
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
        eval_optimized_result=False,
        eval_final_result=True,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
