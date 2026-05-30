import sys
import os
import argparse
import time

sys.path.append('../../../src')
sys.path.append('..')
from attack_paths import get_attack_paths, ALL_TAGS

import torch
from torch import nn
from torchvision.transforms import (
    ToTensor,
    Compose,
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
    TorchvisionClassifierModel,
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import augment_images_fn_generator, Logger, freeze
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.datasets import FaceScrub224
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)


def main(tag):
    paths = get_attack_paths('ppa', tag)

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = paths.cuda_device
    attack_targets = list(range(100))

    sample_batch_size = 10
    optimize_batch_size = 4
    final_selection_batch_size = 4
    evaluation_batch_size = 4
    sample_num = 5000
    optimize_num = 20
    final_num = 5

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

    target_resolution = 224
    eval_resolution = 224

    target_model = auto_classifier_from_pretrained(paths.target_model_ckpt_path)

    eval_model = auto_classifier_from_pretrained(
        paths.eval_model_ckpt_path, register_last_feature_hook=True, weights=None
    )
    # eval_model2 = auto_classifier_from_pretrained(
    #     eval2ckpt_path, register_last_feature_hook=True, weights=None
    # )
    # exit()
    # eval_model = TorchvisionClassifierModel(
    #     'maxvit_t', num_classes=530, weights=None, register_last_feature_hook=True
    # )
    # if not os.path.exists(eval_model_ckpt_path):
    #     print('eval model ckpt not exist')
    # else:
    #     eval_model.load_state_dict(torch.load(eval_model_ckpt_path, map_location='cpu')["state_dict"])
    # exit()

    # print(torch.load(target_model_ckpt_path, map_location='cpu').keys())

    # target_model.load_state_dict(
    #     torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    # )
    # eval_model.load_state_dict(
    #     torch.load(eval_model_ckpt_path, map_location='cpu')['state_dict']
    # )

    mapping = nn.parallel.DataParallel(mapping, device_ids=gpu_devices).to(device)
    target_model = nn.parallel.DataParallel(target_model, device_ids=gpu_devices).to(
        device
    )
    eval_model = nn.parallel.DataParallel(eval_model, device_ids=gpu_devices).to(device)
    # eval_model2 = nn.parallel.DataParallel(eval_model2, device_ids=gpu_devices).to(
    #     device
    # )
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    mapping.eval()
    target_model.eval()
    eval_model.eval()
    # eval_model2.eval()
    generator.eval()

    freeze(mapping)
    freeze(target_model)
    freeze(eval_model)
    # freeze(eval_model2)
    freeze(generator)

    print(target_model)

    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset

    print('load dataset')

    eval_dataset = FaceScrub224(
        paths.eval_dataset_path,
        train=True,
        output_transform=Compose(
            [
                Resize((eval_resolution, eval_resolution)),
                ToTensor(),
                # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        return [img, TF.hflip(img)]

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

    # prepare optimization

    optimize_create_aug_images_fn = augment_images_fn_generator(
        initial_transform=gan_to_target_transform,
        add_origin_image=False,
        augment=RandomResizedCrop(
            (target_resolution, target_resolution),
            scale=(0.9, 1.0),
            ratio=(1.0, 1.0),
            antialias=True,
        ),
        augment_times=1,
    )

    optimization_config = ImageAugmentWhiteBoxOptimizationConfig(
        experiment_dir=paths.experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.005, 'betas': [0.1, 0.1]},
        loss_fn='poincare',
        create_aug_images_fn=optimize_create_aug_images_fn,
        iter_times=70,
        show_loss_info_iters=20,
    )

    optimization_fn = ImageAugmentWhiteBoxOptimization(
        optimization_config, generator, target_model
    )

    # prepare final selection

    final_create_aug_images_fn = augment_images_fn_generator(
        initial_transform=gan_to_target_transform,
        add_origin_image=False,
        augment=Compose(
            [
                RandomResizedCrop(
                    (target_resolution, target_resolution),
                    scale=(0.5, 0.9),
                    ratio=(0.8, 1.2),
                    antialias=True,
                ),
                RandomHorizontalFlip(0.5),
            ]
        ),
        augment_times=100,
    )

    final_select_score_fn = ImageClassificationAugmentConfidence(
        target_model, device=device, create_aug_images_fn=final_create_aug_images_fn
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
        save_individual_res_dir=paths.experiment_dir,
        transform=to_eval_transform,
    )

    # accuracy_metric2 = ImageClassifierAttackAccuracy(
    #     evaluation_batch_size,
    #     eval_model2,
    #     device=device,
    #     description='evaluation-incv3',
    #     transform=to_eval_transform,
    # )

    # distance_metric2 = ImageDistanceMetric(
    #     evaluation_batch_size,
    #     eval_model2,
    #     eval_dataset,
    #     device=device,
    #     description='evaluation-incv3',
    #     save_individual_res_dir=experiment_dir,
    #     transform=to_eval_transform,
    # )

    fid_prdc_metric = ImageFidPRDCMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=paths.experiment_dir,
        fid=True,
        prdc=True,
        transform=to_eval_transform,
    )

    # prepare attack

    face_dist_metric = FaceDistanceMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=paths.experiment_dir,
        transform=to_eval_transform,
    )

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=optimize_num,
        optimize_batch_size=optimize_batch_size,
        optimize_fn=optimization_fn,
        final_num=final_num,
        final_images_score_fn=final_select_score_fn,
        final_select_batch_size=final_selection_batch_size,
        save_dir=paths.experiment_dir,
        save_optimized_images=True,
        save_final_images=True,
        save_kwargs={'normalize': True},
        eval_metrics=[
            accuracy_metric,
            distance_metric,
            face_dist_metric,
            fid_prdc_metric,
            # accuracy_metric2,
            # distance_metric2,
        ],
        eval_optimized_result=False,
        eval_final_result=True,
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)
    # attacker.evaluate_from_pre_generate(generator, attack_targets, 'alt100', device=device)


if __name__ == '__main__':
    # for tag in ['neck50tanh_focal8_lr2e-05_5']:
    #     main(tag)

    import time
    from tqdm import tqdm

    # time.sleep(4200)
    # for i in tqdm(range(3600)):
    #     time.sleep(1)

    # for tag in ['ls0.02']:
    # for tag in ['ls0.005']:
    #     main(tag)
    for tag in ALL_TAGS:
        main(tag)

    import torchvision

    torchvision.models.maxvit_t
