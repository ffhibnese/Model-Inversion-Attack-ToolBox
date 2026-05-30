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
    auto_classifier_from_pretrained, TorchvisionClassifierModel
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,freeze
)
from modelinversion.attack import (
    IntermediateWhiteboxOptimizationConfig,
    StyelGANIntermediateWhiteboxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ImageAugmentClassificationLoss,
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

    os.environ['SAVE_TAG'] = tag

    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'

    device_ids_available = '1'

    experiment_dir = f'./if_resnet152{tag}'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = (
        '/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = '/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/ffhq.pkl'
    target_model_ckpt_path = f'/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224_nopretrain/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    attack_targets = list(range(10))

    sample_batch_size = 20
    optimize_batch_size = 10
    evaluation_batch_size = 4
    sample_num = 5000
    optimize_num = 1

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

    target_resolution = 224
    eval_resolution = 299

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)

    # print(target_model.module.model.fc[-2].__class__.__name__)
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
    generator = nn.parallel.DataParallel(generator, device_ids=gpu_devices).to(device)

    mapping.eval()
    target_model.eval()
    generator.eval()
    
    freeze(target_model)
    freeze(generator)
    freeze(mapping)
    # target_model, eval_model = eval_model, target_model

    # prepare eval dataset



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
        iter_times=[70, 25, 25, 25],
        show_loss_info_iters=50,
    )

    optimization_fn = StyelGANIntermediateWhiteboxOptimization(
        optimization_config, generator, loss_fn
    )

    # prepare metrics



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
            # accuracy_metric,
            # distance_metric,
            # face_dist_metric,
            # fid_prdc_metric,
            # accuracy_metric_2,
            # distance_metric_2,
        ],
        eval_optimized_result=True,
        eval_final_result=False,
        
    )

    attacker = ImageClassifierAttacker(attack_config)

    attacker.attack(attack_targets)

    logger.close()

if __name__ == '__main__':
    tags = [
        # 'ls0.01'
    'ls0.02_scheduler'
        # 'tl0.5_focal3_freemix0.5'
        # 'tl0.5_focal3_ramdomfreemix0.5_cenls0',
        # 'tl0.5_focal3_ramdomfixmix0.5_cenls0'
        # 'ls0.005',
        # 'bido_ih0.05_oh0.5'
        # 'tl0.5_focal3_ramdomfixmix0.5_cenls0.05'
        # 'neck100tanh'
        # 'neck50tanh'
        # 'lora10_neck50tanh'
        # 'neck40tanh'
        # 'lora10end0.9_neck40tanh'
        # 'neck50tanh_focal8_lr0.002_5'

        # 'neck50tanh_focal8_lr2e-05_5',
        # 'tl0.7'
        # 'no'
        # 'tl0.4',
        # 'vib0.005'
        # 'bido_ih0.15_oh1.5'
        # 'ftneck35tanh',
        # 'ftneck40tanh'
        # 'ftneck40tanh_focal8_lr2e-05_5'

        # 'neck50leaky_relu',
        # 'neck50none',
        # 'neck50relu',
        # 'neck50relu6',
        # 'neck50sigmoid',
        # 'neck50tanh',
    ]
    for tag in tags:
        main(tag)