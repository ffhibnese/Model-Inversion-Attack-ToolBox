import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torch.nn import functional as F
from modelinversion.datasets import CelebA
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
    get_stylegan2ata_generator,
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
)


if __name__ == '__main__':

    device_ids_str = '1'
    num_classes = 1000

    experiment_dir = '<fill it>'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = '<fill it>'
    stylegan2ada_ckpt_path = '<fill it>'
    target_model_name = 'resnet152'
    target_model_ckpt_path = '<fill it>'
    eval_model_name = 'inception_v3'
    eval_model_ckpt_path = '<fill it>'
    eval_dataset_path = '<fill it>'
    attack_targets = list(range(100))

    sample_batch_size = 20
    optimize_batch_size = 5
    final_selection_batch_size = 20
    evaluation_batch_size = 50
    sample_num = 5000

    optimize_num = 50
    # final_num = 50

    w_bound_sample_num = 5000
    p_std_ce = 1

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    mapping, generator = get_stylegan2ata_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, single_w=True
    )

    target_resolution = 224
    eval_resolution = 299

    target_model = TorchvisionClassifierModel(
        target_model_name, num_classes=num_classes
    )
    eval_model = TorchvisionClassifierModel(
        eval_model_name,
        num_classes=num_classes,
        resolution=299,
        register_last_feature_hook=True,
    )

    # print(torch.load(target_model_ckpt_path, map_location='cpu').keys())

    target_model.load_state_dict(
        torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    )
    eval_model.load_state_dict(
        torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict']
    )

    mapping = nn.parallel.DataParallel(mapping, device_ids=gpu_devices).to(device)
    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    mapping.eval()
    target_model.eval()
    eval_model.eval()
    generator.eval()

    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset

    eval_dataset = CelebA(
        eval_dataset_path,
        crop_center=False,
        preprocess_resolution=299,
        transform=Compose(
            [
                Resize((eval_resolution, eval_resolution)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    # prepare latent sampler

    w_dim = mapping.module.w_dim

    gan_to_target_transform = Compose(
        [
            CenterCrop((800, 800)),
            Resize((target_resolution, target_resolution), antialias=True),
        ]
    )

    def latent_sampler_aug_fn(img):

        img = gan_to_target_transform(img)
        lower_bound = torch.tensor(-1.0).float().to(img.device)
        upper_bound = torch.tensor(1.0).float().to(img.device)
        img = torch.where(img > upper_bound, upper_bound, img)
        img = torch.where(img < lower_bound, lower_bound, img)
        return [img]

    latents_sampler = ImageAugmentSelectLatentsSampler(
        input_size=w_dim,
        batch_size=sample_batch_size,
        all_sample_num=sample_num,
        generator=generator,
        classifier=target_model,
        device=device,
        latents_mapping=mapping,
        create_aug_images_fn=latent_sampler_aug_fn,
    )

    # prepare min max constrain

    simple_sampler = SimpleLatentsSampler(
        input_size=w_dim, batch_size=sample_batch_size, latents_mapping=mapping
    )

    all_w = simple_sampler([0], w_bound_sample_num)[0]
    all_p = F.leaky_relu(all_w, negative_slope=5)
    all_p_means = torch.mean(all_p, dim=0, keepdim=True)
    all_p_stds = torch.std(all_p, dim=0, keepdim=True, unbiased=False)
    all_p_mins = all_p_means - p_std_ce * all_p_stds
    all_p_maxs = all_p_means + p_std_ce * all_p_stds
    all_w_mins = (
        F.leaky_relu(all_p_mins, negative_slope=0.2)
        .detach()
        .requires_grad_(False)
        .to(device)
    )
    all_w_maxs = (
        F.leaky_relu(all_p_maxs, negative_slope=0.2)
        .detach()
        .requires_grad_(False)
        .to(device)
    )

    latent_constraint = MinMaxConstraint(all_w_mins, all_w_maxs)

    # prepare optimization

    optimize_create_aug_images_fn = augment_images_fn_generator(
        initial_transform=gan_to_target_transform,
        add_origin_image=False,
        augment=RandomHorizontalFlip(),
        augment_times=1,
    )

    optimization_config = ImageAugmentWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.005, 'betas': (0.9, 0.999)},
        loss_fn='ce',
        latent_constraint=latent_constraint,
        create_aug_images_fn=optimize_create_aug_images_fn,
        iter_times=70,
        show_loss_info_iters=5,
    )

    optimization_fn = ImageAugmentWhiteBoxOptimization(
        optimization_config, generator, target_model
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
        save_individual_res_dir=experiment_dir,
        transform=to_eval_transform,
    )

    fid_prdc_metric = ImageFidPRDCMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=experiment_dir,
        fid=True,
        prdc=True,
        transform=to_eval_transform,
    )

    # prepare attack

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=optimize_num,
        optimize_batch_size=optimize_batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
        save_optimized_images=True,
        save_final_images=True,
        save_kwargs={'normalize': True},
        eval_metrics=[accuracy_metric, distance_metric, fid_prdc_metric],
        eval_optimized_result=True,
        eval_final_result=False,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
