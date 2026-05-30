import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    SimpleGenerator64,
    IR152_64,
    FaceNet112,
    auto_classifier_from_pretrained,
)
from modelinversion.sampler import LabelOnlySelectLatentsSampler
from modelinversion.utils import Logger
from modelinversion.attack import (
    BrepOptimizationConfig,
    BrepOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.scores import ImageClassificationAugmentLabelOnlyScore
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)
from modelinversion.datasets import FaceScrub112


for tag in [
    # 'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:

    experiment_dir = f'./results_attack/brep_ir152_{tag}'
    device_ids_available = '1'
    num_classes = 530
    generator_ckpt_path = '../../../checkpoints_v2/gmi/gmi_ffhq64_G.pth'
    discriminator_ckpt_path = '../../../checkpoints_v2/gmi/gmi_ffhq64_D.pth'
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'
    eval_model_ckpt_path = '../../../checkpoints_v2/classifier/facescrub112/facescrub112_facenet112_99.38.pth'
    eval_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    attack_targets = list(range(100))

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

    z_dim = 100

    # target_model = IR152_64(num_classes=num_classes)
    # eval_model = FaceNet112(num_classes, register_last_feature_hook=True)
    generator = SimpleGenerator64(in_dim=z_dim)
    # discriminator = KedmiDiscriminator64(num_classes=num_classes)

    # target_model.load_state_dict(
    #     torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    # )
    # eval_model.load_state_dict(
    #     torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict']
    # )
    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
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

    latents_sampler = LabelOnlySelectLatentsSampler(
        z_dim, batch_size, generator, target_model, device=device
    )

    # prepare eval dataset

    eval_dataset = FaceScrub112(
        eval_dataset_path,
        train=True,
        output_transform=ToTensor(),
    )

    # prepare optimization

    optimization_config = BrepOptimizationConfig(
        experiment_dir=experiment_dir, device=device, iter_times=1000
    )

    image_score_fn = ImageClassificationAugmentLabelOnlyScore(
        classifier=target_model, device=device, correct_score=1, wrong_score=-1
    )

    optimization_fn = BrepOptimization(
        config=optimization_config, generator=generator, image_score_fn=image_score_fn
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
    face_dist_metric = FaceDistanceMetric(
        batch_size,
        eval_dataset,
        device=device,
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
