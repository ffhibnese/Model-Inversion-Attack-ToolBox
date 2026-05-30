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
from modelinversion.datasets import FaceScrub112
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
from modelinversion.sampler import P2ILatentsMixingSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
    MinMaxConstraint,
)
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)


def main(tag):

    paths = get_attack_paths('p2i', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device

    attack_targets = list(range(100))

    sample_batch_size = 40
    optimize_batch_size = 30
    final_selection_batch_size = 30
    evaluation_batch_size = 50
    sample_num = 5000

    optimize_num = 10

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

    mapping, generator = get_stylegan2ada_generator(
        paths.stylegan2ada_path, paths.stylegan2ada_ckpt_path, single_w=True
    )

    target_resolution = 64
    eval_resolution = 112

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path,
        register_last_feature_hook=True,
    )
    p2i_adapter = auto_adapter_from_pretrained(
        os.path.join(paths.p2i_adapter_path, 'p2i_adapter.pth')
    )

    mapping = nn.parallel.DataParallel(mapping, device_ids=gpu_devices).to(device)
    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)
    p2i_adapter = nn.parallel.DataParallel(p2i_adapter, device_ids=gpu_devices).to(
        device
    )

    mapping.eval()
    target_model.eval()
    eval_model.eval()
    generator.eval()
    p2i_adapter.eval()

    # prepare eval dataset

    eval_dataset = FaceScrub112(
        paths.eval_dataset_path,
        train=True,
        output_transform=Compose(
            [
                ToTensor(),
            ]
        ),
    )

    # prepare latent sampler

    w_dim = mapping.module.w_dim

    gan_to_target_transform = Compose(
        [
            CenterCrop((176, 176)),
            Resize((target_resolution, target_resolution), antialias=True),
        ]
    )

    from torchvision.datasets import ImageFolder

    public_dataset = ImageFolder(
        paths.lomma_public_dataset_path,
        transform=Compose(
            [
                Resize((256, 256)),
                ToTensor(),
            ]
        ),
    )

    latents_sampler = P2ILatentsMixingSampler(
        input_size=w_dim,
        batch_size=sample_batch_size,
        avg_num=10,
        classifier=target_model,
        generator=generator,
        p2i_adapter=p2i_adapter,
        output_bias=torch.load(
            os.path.join(paths.p2i_adapter_path, 'avg_w.pth'), map_location='cpu'
        ),
        dataset=public_dataset,
    )

    optimization_config = ImageAugmentWhiteBoxOptimizationConfig(
        experiment_dir=paths.experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.005, 'betas': (0.9, 0.999)},
        loss_fn='ce',
        latent_constraint=None,
        create_aug_images_fn=None,
        iter_times=0,
        show_loss_info_iters=5,
    )

    optimization_fn = ImageAugmentWhiteBoxOptimization(
        optimization_config, generator, target_model
    )

    # prepare metrics

    to_eval_transform = Compose(
        [
            CenterCrop((176, 176)),
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
        optimize_batch_size=optimize_batch_size,
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


if __name__ == '__main__':
    for tag in ALL_TAGS:
        main(tag)
