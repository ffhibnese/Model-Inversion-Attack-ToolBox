import sys
import os
import time

sys.path.append("../../../src")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop

from modelinversion.models import (
    SimpleGenerator64,
    GmiDiscriminator64,
    auto_classifier_from_pretrained,
    get_stylegan2ada_generator,
)
from modelinversion.models.adapters import P2IConfidence2StyleAdapter
from modelinversion.train import GmiGanTrainer, GmiGanTrainConfig, train_p2i_adapter
from modelinversion.utils import Logger
from modelinversion.utils.losses import LpipsLoss, ArcfaceIDCosineLoss, LandmarkLoss
from modelinversion.datasets import GeneratorDataset
from modelinversion.sampler import SimpleLatentsSampler


def main(tag):

    num_classes = 530

    target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'
    arcface_ckpt_path = (
        "/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/p2i/arcface.pth"
    )
    parsing_model_ckpt_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/p2i/parsing_model.pth'
    )
    real_dataset_path = f'./real_dataset/ffhq64_facescrub64_ir152_{tag}_dataset'
    fake_dataset_path = (
        f'./fake_dataset/ffhq64_facescrub64_ir152__{tag}_dataset/dataset.pt'
    )

    dataset_map_name = 'ffhq64_facescrub64'
    target_name = f'ir152_{tag}'
    stylegan2ada_path = (
        '/mnt/data/<usrname>/mywork/lora_defense/test_resp/stylegan2-ada-pytorch'
    )
    stylegan2ada_ckpt_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/stylegan2ada/stylegan2-ffhq-256x256.pkl'
    experiment_dir = f'./results_adapter2/{dataset_map_name}/{target_name}'

    batch_size = 8

    device_ids_str = '1'

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path)
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model.eval()

    mapping, generator = get_stylegan2ada_generator(
        stylegan2ada_path, stylegan2ada_ckpt_path, optimize_in_w=True
    )

    mapping = mapping.to(device).eval()
    generator = generator.to(device).eval()

    lpips_loss_fn = LpipsLoss(device=device, scale=1.0)
    arcface_loss_fn = ArcfaceIDCosineLoss(
        ckpt_path=arcface_ckpt_path, device=device, scale=1
    )

    landmark_loss_fn = LandmarkLoss(
        ckpt_path=parsing_model_ckpt_path, device=device, scale=1
    )

    # print(target_model.training)
    # exit()

    # prepare dataset

    from torchvision.datasets import ImageFolder

    real_dataset = ImageFolder(
        real_dataset_path,
        transform=Compose(
            [
                Resize((256, 256)),
                ToTensor(),
            ]
        ),
    )
    # dataset = CelebA64(dataset_path, ToTensor())
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # sampler=InfiniteSamplerWrapper(dataset),
    )

    fake_dataset = GeneratorDataset.from_precreate(
        fake_dataset_path,
        generator=get_stylegan2ada_generator(
            stylegan2ada_path, stylegan2ada_ckpt_path, optimize_in_w=False
        )[1]
        .to(device)
        .eval(),
        device=device,
        transform=Compose(
            [
                CenterCrop(176),
                Resize((256, 256)),
            ]
        ),
    )

    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=fake_dataset.collate_fn_for_p2i,
    )

    p2i_apdater = P2IConfidence2StyleAdapter(
        num_classes=num_classes,
        n_styles=14,
        arcface_model_path=arcface_ckpt_path,
    )
    p2i_apdater = p2i_apdater.to(device).train()

    # prepare min max constrain

    with torch.no_grad():
        simple_sampler = SimpleLatentsSampler(
            input_size=512, batch_size=batch_size, latents_mapping=mapping
        )

        all_w = simple_sampler([0], 5000)[0]
        # all_p = nn.functional.leaky_relu(all_w, negative_slope=5)
        all_w_means = torch.mean(all_w, dim=0, keepdim=True).to(device)

    # mapping = C2fThreeLayerMlpOutputMapping(
    #     target_model.module.num_classes, 4096, embed_model.module.num_classes
    # )
    # mapping = nn.DataParallel(mapping).to(device)
    # mapping.train()

    optimizer = torch.optim.Adam(p2i_apdater.parameters(), lr=0.001)
    optim_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.99
    )

    train_p2i_adapter(
        40,
        target_model=target_model,
        p2i_adapter=p2i_apdater,
        optimizer=optimizer,
        stylegan_mapping=mapping,
        stylegan=generator,
        avg_w=all_w_means,
        real_dataloader=real_dataloader,
        fake_dataloader=fake_dataloader,
        device=device,
        save_path=experiment_dir,
        lpips_loss_fn=lpips_loss_fn,
        arcface_loss_fn=arcface_loss_fn,
        landmark_loss_fn=landmark_loss_fn,
        schedular=optim_scheduler,
    )

    logger.close()


for tag in [
    'no',
    'vib0.01',
    'bido0.01_0.1_pretrain',
    'ls0.05',
    'tl0.5',
    'rolss0.0_2',
]:
    main(tag)
