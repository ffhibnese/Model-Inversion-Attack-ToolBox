import sys
import os
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
from modelinversion.sampler import (
    LayeredFlowLatentsSampler,
    GaussianMixtureLatentsSampler,
    FlowConfig,
)
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,
)
from modelinversion.attack import (
    MinerWhiteBoxOptimizationConfig,
    MinerWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    VmiLoss,
    VmiTrainer,
    VmiAttacker,
)
from modelinversion.datasets import CelebA112
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
)

if __name__ == '__main__':

    device_ids_available = '0'
    num_classes = 1000

    experiment_dir = '/data/qyx/Model-Inversion-Attack-ToolBox/test/vmi'
    """Download stylegan2-ada from https://github.com/NVlabs/stylegan2-ada-pytorch and record the file path as 'stylegan2ada_path' 
    """
    stylegan2ada_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/stylegan2_ada'
    stylegan2ada_ckpt_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/neurips2021-celeba-stylegan/network-snapshot-002298.pkl'
    target_model_name = 'ir152_64'
    target_model_ckpt_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/celeba64/celeba64_ir152_93.71.pth'
    eval_model_name = 'facenet112'
    eval_model_ckpt_path = '/data/qyx/Model-Inversion-Attack-ToolBox/test/celeba112/celeba112_facenet112_95.88.pth'
    eval_dataset_path = (
        '/data/qyx/Model-Inversion-Attack-ToolBox/test/celeba/private_train'
    )
    attack_targets = list(range(100))

    sample_batch_size = 16
    evaluation_batch_size = 50
    train_epochs = 30
    
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

    # prepare eval dataset

    eval_dataset = CelebA112(
        eval_dataset_path,
        output_transform=ToTensor(),
    )

    # prepare latent sampler

    w_dim = mapping.module.w_dim
    z_dim = mapping.module.z_dim
    num_ws = mapping.module.num_ws

    # prepare flow params
    flow_params = FlowConfig(
        k=z_dim,
        l=num_ws,
        flow_permutation='shuffle',
        flow_K=10,
        flow_glow=True,
        flow_coupling='additive',
        flow_L=3,
        flow_use_actnorm=True,
    )

    # prepare optimization

    optimization_config = MinerWhiteBoxOptimizationConfig(
        experiment_dir=experiment_dir,
        device=device,
        optimizer='SGD',
        optimizer_kwargs={'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 0},
        iter_times=150,
        show_loss_info_iters=10,
        batch_size=sample_batch_size,
    )

    trainer = VmiTrainer(
        epochs=train_epochs,
        experiment_dir=experiment_dir,
        input_size=w_dim,
        batch_size=sample_batch_size,
        generator=generator,
        flow_params=flow_params,
        device=device,
        latents_mapping=mapping,
        classifier=target_model,
        loss_weights={
            'lambda_attack': 1.0,
            'lambda_miner_entropy': 0.0,
            'lambda_kl': 1e-3,
        },
        optimize_config=optimization_config,
    )

    trainer.train_miners(cores=3, targets=attack_targets, root_path=experiment_dir)

    # prepare metrics

    accuracy_metric = ImageClassifierAttackAccuracy(
        evaluation_batch_size,
        eval_model,
        device=device,
        description='evaluation',
        transform=None,
    )

    distance_metric = ImageDistanceMetric(
        evaluation_batch_size,
        eval_model,
        eval_dataset,
        device=device,
        description='evaluation',
        save_individual_res_dir=experiment_dir,
        transform=None,
    )

    fid_prdc_metric = ImageFidPRDCMetric(
        evaluation_batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=experiment_dir,
        fid=True,
        prdc=True,
        transform=None,
    )

    # prepare attack

    # attack_config = ImageClassifierAttackConfig(
    #     latent_sampler,
    #     optimize_num=optimize_num,
    #     optimize_batch_size=optimize_batch_size,
    #     optimize_fn=optimization_fn,
    #     save_dir=experiment_dir,
    #     save_optimized_images=True,
    #     save_final_images=False,
    #     save_kwargs={'normalize': True},
    #     eval_metrics=[accuracy_metric, distance_metric, fid_prdc_metric],
    #     eval_optimized_result=False,
    #     eval_final_result=True,
    # )

    # attacker = ImageClassifierAttacker(attack_config)

    # attacker.attack(attack_targets)
