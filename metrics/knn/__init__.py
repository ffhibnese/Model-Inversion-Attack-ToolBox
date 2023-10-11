import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms


def get_private_feats(E, private_feats_path, private_loader, resolution=112):
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
        images = augmentation.Resize((resolution, resolution))(images)
        feats = E(images).result

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