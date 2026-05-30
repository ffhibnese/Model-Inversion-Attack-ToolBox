import sys
import os
import time

sys.path.append('../../../src')

import kornia
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from modelinversion.models import (
    PlgmiGenerator256,
    PlgmiDiscriminator256,
    auto_classifier_from_pretrained,
)
from modelinversion.train import PlgmiGanTrainer, PlgmiGanTrainConfig
from modelinversion.utils import Logger, freeze
from modelinversion.datasets import InfiniteSamplerWrapper, LabelImageFolder


import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import auto_classifier_from_pretrained
from modelinversion.datasets import (
    top_k_selection,
    preprocess_celeba_fn,
    preprocess_facescrub_fn,
)

if __name__ == '__main__':

    top_k = 30
    num_classes = 530
    loradim = 1
    tag = ''
    target_model_ckpt_path = f'/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    src_dataset_path = '/mnt/data/<usrname>/datasets/ffhq256/'
    tag = f'{tag}'
    dst_dataset_path = f'./dataset/plg_ffhq256_facescrub256_resnet152{tag}_dataset'
    experiment_dir = f'./results_gan/plg_ffhq256_facescrub256_resnet152{tag}_gan'

    dataset_path = dst_dataset_path

    batch_size = 32
    device_ids_str = '7'

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
    # target_model = nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.eval()

    # dataset generation

    transform = TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if not os.path.exists(dst_dataset_path):

        top_k_selection(
            top_k=top_k,
            src_dataset_path=src_dataset_path,
            dst_dataset_path=dst_dataset_path,
            batch_size=batch_size,
            target_model=target_model,
            num_classes=num_classes,
            device=device,
            create_aug_images_fn=lambda img: [transform(img)],
        )

    batch_size = 32
    max_iters = 100000

    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img

    dataset = LabelImageFolder(
        dataset_path,
        transform=Compose([ToTensor(), _noise_adder]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=8,
    )

    # prepare GANs

    z_dim = 128

    generator = PlgmiGenerator256(num_classes, dim_z=z_dim).to(device)
    discriminator = PlgmiDiscriminator256(num_classes).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9)
    )

    # prepare trainer

    data_augment = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop(
            (256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)
        ),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    train_configs = PlgmiGanTrainConfig(
        experiment_dir=experiment_dir,
        # train args
        batch_size=batch_size,
        input_size=z_dim,
        generator=generator,
        discriminator=discriminator,
        num_classes=num_classes,
        target_model=target_model,
        classification_loss_fn='max_margin',
        device=device,
        augment=data_augment,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        # log args
        save_ckpt_iters=1000,
        show_images_iters=1000,
        show_train_info_iters=100,
    )

    # train gan

    trainer = PlgmiGanTrainer(train_configs)

    freeze(target_model)

    trainer.train(dataloader, max_iters)
