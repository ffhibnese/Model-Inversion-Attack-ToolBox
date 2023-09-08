import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms

import inception
import metrics.fid
import utils


def calc_fid(recovery_img_path, private_img_path, batch_size=64):
    """
    Calculate the FID of the reconstructed image.
    :param recovery_img_path: the dir of reconstructed images
    :param private_img_path: the dir of private data
    :param batch_size: batch size
    :return: FID of reconstructed images
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inception_model = inception.InceptionV3().to(device)

    recovery_list, private_list = [], []

    # get the reconstructed images
    list_of_idx = os.listdir(recovery_img_path)  # [0,1,2,3,4,5....]
    if len(list_of_idx) == 0:
        return -1000
    for idx in list_of_idx:
        success_recovery_num = len(os.listdir(os.path.join(recovery_img_path, idx)))
        for recovery_img in os.listdir(os.path.join(recovery_img_path, idx)):
            image = Image.open(os.path.join(recovery_img_path, idx, recovery_img))
            image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
            recovery_list.append(image.numpy())
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

    mu_fake, sigma_fake = metrics.fid.calculate_activation_statistics(
        recovery_images, inception_model, batch_size, device=device
    )
    mu_real, sigma_real = metrics.fid.calculate_activation_statistics(
        private_images, inception_model, batch_size, device=device
    )
    fid_score = metrics.fid.calculate_frechet_distance(
        mu_fake, sigma_fake, mu_real, sigma_real
    )

    return fid_score


def calc_knn(feat, iden, path):
    """
    Get the KNN Dist from reconstructed images to private date
    :param feat: features of reconstructed images output by evaluation model
    :param iden: target class
    :param path: the filepath of the private features
    :return: KNN Distance
    """
    iden = iden.cpu().long()
    feat = feat.cpu()
    true_feat = torch.from_numpy(np.load(os.path.join(path, "private_feats.npy"))).float()
    info = torch.from_numpy(np.load(os.path.join(path, "private_targets.npy"))).view(-1).long()
    bs = feat.size(0)
    tot = true_feat.size(0)
    knn_dist = 0
    for i in range(bs):

        knn = 1e8
        for j in range(tot):
            if info[j] == iden[i]:  # 在private domain中找对应类别的图片
                dist = torch.sum((feat[i, :] - true_feat[j, :]) ** 2)  # 计算特征的l2距离
                if dist < knn:
                    knn = dist
        knn_dist += knn

    return (knn_dist / bs).item()


def get_knn_dist(E, infered_image_path, private_feats_path):
    """
    Get KNN Dist of reconstructed images.
    :param E:
    :param infered_image_path:
    :param private_feats_path:
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    list_of_idx = os.listdir(infered_image_path)

    images_list = []
    targets_list = []
    # load reconstructed images
    for idx in list_of_idx:
        for filename in os.listdir(os.path.join(infered_image_path, idx)):
            target, seed = os.path.splitext(filename)[0].strip().split('_')[-2:]
            image = Image.open(os.path.join(infered_image_path, idx, filename))
            image = transforms.functional.to_tensor(image)
            images_list.append(image)
            targets_list.append(int(target))

    images = torch.stack(images_list, dim=0)
    targets = torch.LongTensor(targets_list)
    # get features of reconstructed images
    infered_feats = None
    images_spilt_list = images.chunk(int(images.shape[0] / 10))
    for i, images in enumerate(images_spilt_list):
        images = augmentation.Resize((112, 112))(images).to(device)
        feats = E(images)[0]
        if i == 0:
            infered_feats = feats.detach().cpu()
        else:
            infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)
    # calc the knn dist
    knn_dist = calc_knn(infered_feats, targets, private_feats_path)

    return knn_dist
