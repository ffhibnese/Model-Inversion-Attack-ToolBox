import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms

from . import fid_utils
from . import inceptionv3
# from ...models import 


def calc_fid(recovery_img_path, private_img_path, batch_size=64):
    """
    Calculate the FID of the reconstructed image.
    :param recovery_img_path: the dir of reconstructed images
    :param private_img_path: the dir of private data
    :param batch_size: batch size
    :return: FID of reconstructed images
    
    recovery_imgs files: label / imgs
        like
            0 / a.img
            1 / b.img
            2 / c.img
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inception_model = inceptionv3.InceptionV3().to(device)

    recovery_list, private_list = [], []

    # get the reconstructed images
    # list_of_idx = os.listdir(recovery_img_path)  # [0,1,2,3,4,5....]
    # if len(list_of_idx) == 0:
    #     return -1000
    # for idx in list_of_idx:
    #     success_recovery_num = len(os.listdir(os.path.join(recovery_img_path, idx)))
    #     for recovery_img in os.listdir(os.path.join(recovery_img_path, idx)):
    #         image = Image.open(os.path.join(recovery_img_path, idx, recovery_img))
    #         image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    #         recovery_list.append(image.numpy())
    
    recovery_loader = iter(torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            recovery_img_path,
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ), batch_size, )
    )
    for imgs, _ in recovery_loader:
        recovery_list.append(imgs.numpy())
    # get real images from private date
    eval_loader = iter(torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            private_img_path,
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ), batch_size, )
    )
    for imgs, _ in eval_loader:
        private_list.append(imgs.numpy())

    recovery_images = np.concatenate(recovery_list)
    private_images = np.concatenate(private_list)

    mu_fake, sigma_fake = fid_utils.calculate_activation_statistics(
        recovery_images, inception_model, batch_size, device=device
    )
    mu_real, sigma_real = fid_utils.calculate_activation_statistics(
        private_images, inception_model, batch_size, device=device
    )
    fid_score = fid_utils.calculate_frechet_distance(
        mu_fake, sigma_fake, mu_real, sigma_real
    )

    return fid_score