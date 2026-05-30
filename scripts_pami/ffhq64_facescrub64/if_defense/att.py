import sys
import os
import argparse
import time

sys.path.append('../../../src')

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
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
)
from modelinversion.attack import (
    IntermediateWhiteboxOptimizationConfig,
    StyelGANIntermediateWhiteboxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ImageAugmentClassificationLoss,
)
from modelinversion.datasets import FaceScrub112
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)


def main(tag):

    device_ids_available = '7'

    experiment_dir = f'./if_ir152_{tag}'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/stylegan2-ffhq-256x256.pkl'
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'
    # '/mnt/data/<usrname>/Model-Inversion-Attack-ToolBox/results/train_facescrub64_ir152_lora/facescrub64_ir152_lora.pth'
    eval_model_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/facescrub112/facescrub112_facenet112_99.38.pth'
    # eval_model_ckpt_path_2 = '/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_maxvit_t/facescrub224_maxvit_t.pth'
    eval_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'
    attack_targets = list(range(100))

    sample_batch_size = 20
    optimize_batch_size = 8
    final_selection_batch_size = 8
    evaluation_batch_size = 4
    sample_num = 5000
    optimize_num = 10

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'attack_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare models

    mapping, generator = get_stylegan2ada_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, single_w=True
    )

    target_resolution = 64
    eval_resolution = 112

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    eval_model = auto_classifier_from_pretrained(
        eval_model_ckpt_path, register_last_feature_hook=True
    )

    print(target_model)

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
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    mapping.eval()
    target_model.eval()
    eval_model.eval()
    generator.eval()

    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset

    eval_dataset = FaceScrub112(
        eval_dataset_path,
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
            CenterCrop((176, 176)),
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

    loss_fn = ImageAugmentClassificationLoss(
        target_model,
        loss_fn='poincare',
        create_aug_images_fn=optimize_create_aug_images_fn,
    )

    optimization_config = IntermediateWhiteboxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.005, 'betas': [0.1, 0.1]},
        iter_times=[600, 100, 100],
        show_loss_info_iters=50,
    )

    optimization_fn = StyelGANIntermediateWhiteboxOptimization(
        optimization_config, generator, loss_fn
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

    face_dist_metric = FaceDistanceMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_res_dir=experiment_dir,
        transform=to_eval_transform,
    )

    attack_config = ImageClassifierAttackConfig(
        latents_sampler,
        optimize_num=optimize_num,
        optimize_batch_size=optimize_batch_size,
        optimize_fn=optimization_fn,
        save_dir=experiment_dir,
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
    for tag in [
        'no',
        'vib0.01',
        'bido0.01_0.1_pretrain',
        'ls0.05',
        'tl0.5',
        'rolss0.0_2',
    ]:
        main(tag)
