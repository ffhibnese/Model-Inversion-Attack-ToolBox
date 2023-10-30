import numpy as np
import os
import torch
import torchvision
from PIL import Image
from kornia import augmentation
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import collections
from torchvision.transforms import ToTensor

def generate_private_feats(eval_model, img_dir, save_dir, transforms=None, batch_size=60, device='cpu', exist_ignore=False):
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    if len(os.listdir(save_dir)) and exist_ignore > 2:
        return
    
    results = collections.defaultdict(list)
    dataset = ImageFolder(img_dir, transform=ToTensor())
    data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for imgs, labels in data_loaders:
        if transforms is not None:
            imgs = transforms(imgs)
        
        imgs = imgs.to(device)
        
        feat = eval_model(imgs).feat[-1].cpu()
        for i in range(len(labels)):
            # save_dir = os.path.join(save_dir, f'{labels[i].item()}')
            # os.makedirs(save_dir)
            label = labels[i].item()
            results[label].append(feat[i].detach().cpu().unsqueeze(0).numpy())
            
    for label, feats in results.items():
        feats = np.concatenate(feats, axis=0)
        np.save(os.path.join(save_dir, f'{label}.npy'), feats)

def calc_knn(fake_feat_dir, private_feat_dir):
    fake_feat_files = os.listdir(fake_feat_dir)
    
    total_knn = 0
    total_num = 0
    
    for fake_feat_file in fake_feat_files:
        fake_path = os.path.join(fake_feat_dir, fake_feat_file)
        private_path = os.path.join(private_feat_dir, fake_feat_file)
        if not os.path.exists(private_path):
            continue
        
        # (N_f, dim)
        fake_feat = np.load(fake_path)[:, None, :]
        # (N_p, dim)
        private_feat = np.load(private_path)[None, :, :]
        # (N_f, N_p)
        diff = ((fake_feat - private_feat) ** 2).sum(axis=-1)
        
        knns = np.min(diff, axis=1)
        total_knn += knns.sum()
        total_num += len(knns)
    if total_num == 0:
        raise RuntimeError('NO feat file for fake or private')
    
    return total_knn / total_num
        
    
# def calc_knn(feat, iden, private_feats_path):
#     """
#     Get the KNN Dist from reconstructed images to private date
#     :param feat: features of reconstructed images output by evaluation model
#     :param iden: target class
#     :param path: the filepath of the private features
#     :return: KNN Distance
#     """
#     iden = iden.cpu().long()
#     feat = feat.cpu()
#     true_feat = torch.from_numpy(np.load(os.path.join(private_feats_path, "private_feats.npy"))).float()
#     info = torch.from_numpy(np.load(os.path.join(private_feats_path, "private_targets.npy"))).view(-1).long()
#     bs = feat.size(0)
#     tot = true_feat.size(0)
#     knn_dist = 0
#     for i in range(bs):

#         knn = 1e8
#         for j in range(tot):
#             if info[j] == iden[i]:  # 在private domain中找对应类别的图片
#                 dist = torch.sum((feat[i, :] - true_feat[j, :]) ** 2)  # 计算特征的l2距离
#                 if dist < knn:
#                     knn = dist
#         knn_dist += knn

#     return (knn_dist / bs).item()

# def get_knn_dist(E, infered_image_path, private_feats_path, resolution, device='cpu'):
#     """
#     Get KNN Dist of reconstructed images.
#     :param E:
#     :param infered_image_path:
#     :param private_feats_path:
#     :return:
#     """
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     list_of_idx = os.listdir(infered_image_path)
#     list_of_idx = [d for d in list_of_idx if os.path.isdir(os.path.join(infered_image_path, d))]

#     images_list = []
#     targets_list = []
#     # load reconstructed images
#     for idx in list_of_idx:
#         for filename in os.listdir(os.path.join(infered_image_path, idx)):
#             # target, seed = os.path.splitext(filename)[0].strip().split('_')[-2:]
#             if filename.endswith(('jpg', 'jpeg', 'png')):
#                 target = idx
#                 image = Image.open(os.path.join(infered_image_path, idx, filename))
#                 image = transforms.functional.to_tensor(image)
#                 images_list.append(image)
#                 targets_list.append(int(target))

#     images = torch.stack(images_list, dim=0)
#     targets = torch.LongTensor(targets_list)
#     # get features of reconstructed images
#     infered_feats = None
#     images_spilt_list = images.chunk(int(images.shape[0] / 10))
#     for i, images in enumerate(images_spilt_list):
#         images = augmentation.Resize((resolution, resolution))(images).to(device)
#         feats = E(images).feat[-1]
#         if i == 0:
#             infered_feats = feats.detach().cpu()
#         else:
#             infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)
#     # calc the knn dist
#     knn_dist = calc_knn(infered_feats, targets, private_feats_path)

#     return knn_dist