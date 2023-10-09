import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms

from .metrics import fid as fid_utils
from . import utils
from .models import inception
from models import *


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


def get_private_feats(E, private_feats_path, private_loader):
    """
    Get the features of private data on the evaluation model, and save as file.
    :param E: Evaluation model
    :param private_feats_path: save path
    :param private_loader: dataloader of the private data
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(private_feats_path):
        os.makedirs(private_feats_path)

    private_feats = None
    private_targets = None
    for i, (images, targets) in enumerate(private_loader):
        images, targets = images.to(device), targets.to(device)

        targets = targets.view(-1)
        images = augmentation.Resize((112, 112))(images)
        feats = E(images)[0]

        if i == 0:
            private_feats = feats.detach().cpu()
            private_targets = targets.detach().cpu()
        else:
            private_feats = torch.cat([private_feats, feats.detach().cpu()], dim=0)
            private_targets = torch.cat([private_targets, targets.detach().cpu()], dim=0)

        print("private_feats: ", private_feats.shape)
        print("private_targets: ", private_targets.shape)

    np.save(os.path.join(private_feats_path, 'private_feats.npy'), private_feats.numpy())
    np.save(os.path.join(private_feats_path, 'private_targets.npy'), private_targets.numpy())
    print("Done!")


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


def evaluate(args, current_iter, gen, device, inception_model=None, eval_iter=None):
    """Evaluate in the training process."""
    calc_fid = (inception_model is not None) and (eval_iter is not None)
    num_batches = args.n_eval_batches
    gen.eval()
    fake_list, real_list = [], []
    conditional = args.cGAN
    for i in range(0, num_batches):
        if conditional:
            class_id = i % args.num_classes
        else:
            class_id = None
        fake = utils.generate_images(
            gen, device, args.batch_size, args.gen_dim_z,
            args.gen_distribution, class_id=class_id
        )
        if calc_fid and i <= args.n_fid_batches:
            fake_list.append((fake.cpu().numpy() + 1.0) / 2.0)
            real_list.append((next(eval_iter)[0].cpu().numpy() + 1.0) / 2.0)
        # Save generated images.
        root = args.eval_image_root
        if conditional:
            root = os.path.join(root, "class_id_{:04d}".format(i))
        if not os.path.isdir(root):
            os.makedirs(root)
        fn = "image_iter_{:07d}_batch_{:04d}.png".format(current_iter, i)
        torchvision.utils.save_image(
            fake, os.path.join(root, fn), nrow=4, normalize=True, scale_each=True
        )
    # Calculate FID scores
    if calc_fid:
        fake_images = np.concatenate(fake_list)
        real_images = np.concatenate(real_list)
        mu_fake, sigma_fake = fid_utils.calculate_activation_statistics(
            fake_images, inception_model, args.batch_size, device=device
        )
        mu_real, sigma_real = fid_utils.calculate_activation_statistics(
            real_images, inception_model, args.batch_size, device=device
        )
        fid_score = fid_utils.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
    else:
        fid_score = -1000
    gen.train()
    return fid_score
