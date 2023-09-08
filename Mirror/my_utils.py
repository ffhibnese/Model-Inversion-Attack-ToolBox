#!/usr/bin/env python3
# coding=utf-8
from collections import defaultdict
from itertools import chain, combinations
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as F


ALL_MEANS = {
    'vgg16': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_5class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_8class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_9class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_16class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_24class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_10class_dp_sgd': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_10class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16bn': [131.45376586914062, 103.98748016357422, 91.46234893798828],
    'resnet50': [131.0912, 103.8827, 91.4953],
    'inception_resnetv1_vggface2': [127.5, 127.5, 127.5],
    'inception_resnetv1_vggface2_8631': [127.5, 127.5, 127.5],
    'inception_resnetv1_casia': [127.5, 127.5, 127.5],
    'sphere20a': [127.5, 127.5, 127.5],
    'ccs19ami_facescrub': [0., 0., 0.],
    'ccs19ami_facescrub_rgb': [0., 0., 0.],
    'azure': [0., 0., 0.],
    'cat_resnet18': [0.485*255, 0.456*255, 0.406*255],
    'resnet18_10class': [0.485*255, 0.456*255, 0.406*255],
    'car_resnet34': [0.5*255, 0.5*255, 0.5*255],
    'resnet50_8631': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
    'resnet50_8631_adv': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
    'resnet50_8631_vib': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
    'resnet50_100': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
    'resnet50_100_adv': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
    'resnet50_100_vib': [91.4953, 103.8827, 131.0912],  # Note: BGR!!
}
ALL_STDS = {
    'vgg16': [1., 1., 1.],
    'vgg16_5class': [1., 1., 1.],
    'vgg16_8class': [1., 1., 1.],
    'vgg16_9class': [1., 1., 1.],
    'vgg16_16class': [1., 1., 1.],
    'vgg16_24class': [1., 1., 1.],
    'vgg16_10class_dp_sgd': [1., 1., 1.],
    'vgg16_10class': [1., 1., 1.],
    'vgg16bn': [1., 1., 1.],
    'resnet50': [1., 1., 1.],
    'inception_resnetv1_vggface2': [128.0, 128.0, 128.0],
    'inception_resnetv1_vggface2_8631': [128.0, 128.0, 128.0],
    'inception_resnetv1_casia': [128.0, 128.0, 128.0],
    'sphere20a': [128.0, 128.0, 128.0],
    'ccs19ami_facescrub': [255., 255., 255.],
    'ccs19ami_facescrub_rgb': [255., 255., 255.],
    'azure': [255., 255., 255.],
    'cat_resnet18': [0.229*255, 0.224*255, 0.225*255],
    'resnet18_10class': [0.229*255, 0.224*255, 0.225*255],
    'car_resnet34': [0.5*255, 0.5*255, 0.5*255],
    'resnet50_8631': [1., 1., 1.],
    'resnet50_8631_adv': [1., 1., 1.],
    'resnet50_8631_vib': [1., 1., 1.],
    'resnet50_100': [1., 1., 1.],
    'resnet50_100_adv': [1., 1., 1.],
    'resnet50_100_vib': [1., 1., 1.],
}


def denormalize(image_tensor, arch_name):
    """
    output image is in [0., 1.] and RGB channel
    """
    std = ALL_STDS[arch_name]
    mean = ALL_MEANS[arch_name]
    image_tensor = image_tensor * torch.tensor(std, device=image_tensor.device)[:, None, None] + torch.tensor(mean, device=image_tensor.device)[:, None, None]
    image_tensor = image_tensor / 255.

    if 'resnet50_100' in arch_name or 'resnet50_8631' in arch_name:
        # change BGR to RGB
        if image_tensor.ndim == 4:
            assert image_tensor.shape[1] == 3
            image_tensor = image_tensor[:, [2, 1, 0]]
        else:
            assert image_tensor.ndim == 3
            assert image_tensor.shape[0] == 3
            image_tensor = image_tensor[[2, 1, 0]]

    return torch.clamp(image_tensor, 0., 1.)


def normalize(image_tensor, arch_name):
    """
    input image is in [0., 255.] and RGB channel
    """
    if 'resnet50_100' in arch_name or 'resnet50_8631' in arch_name:
        # change RGB to BGR
        if image_tensor.ndim == 4:
            assert image_tensor.shape[1] == 3
            image_tensor = image_tensor[:, [2, 1, 0]]
        else:
            assert image_tensor.ndim == 3
            assert image_tensor.shape[0] == 3
            image_tensor = image_tensor[[2, 1, 0]]
    std = ALL_STDS[arch_name]
    mean = ALL_MEANS[arch_name]
    image_tensor = (image_tensor-torch.tensor(mean, device=image_tensor.device)[:, None, None])/torch.tensor(std, device=image_tensor.device)[:, None, None]
    return image_tensor


def crop_img_for_sphereface(img):
    assert len(img.shape) == 3 or len(img.shape) == 4
    # resize the img to 256 first because the following crop area are defined in 256 scale
    img = F.resize(img, (256, 256))
    return img[..., 16:226, 38:218]


def crop_img_for_ccs19ami(img):
    raise AssertionError('do not use this')
    assert len(img.shape) == 3 or len(img.shape) == 4
    # resize the img to 256 first because the following crop area are defined in 256 scale
    img = F.resize(img, (256, 256))
    return img[..., 34:214, 40:220]


def crop_img(img, arch_name):
    if arch_name == 'sphere20a':
        return crop_img_for_sphereface(img)
    elif arch_name.startswith('ccs19ami'):
        raise AssertionError('do not use white-box attack for ccs19ami')
        return crop_img_for_ccs19ami(img)
    else:
        return img


def resize_img(img, image_resolution):
    if not isinstance(image_resolution, tuple):
        image_resolution = (image_resolution, image_resolution)
    return F.resize(img, image_resolution)


def clip(image_tensor, use_fp16, arch_name):
    assert not use_fp16
    std = ALL_STDS[arch_name]
    mean = ALL_MEANS[arch_name]
    mean = torch.tensor(mean, device=image_tensor.device)[:, None, None]
    std = torch.tensor(std, device=image_tensor.device)[:, None, None]
    image_tensor = image_tensor.detach().clone()*std + mean
    return (torch.clamp(image_tensor, 0., 255.) - mean) / std


def clip_quantile_bound(inputs, all_mins, all_maxs):
    clipped = torch.max(torch.min(inputs, all_maxs), all_mins)
    return clipped


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))


def compute_topk_labels(logits, k):
    topk_conf, topk_ind = torch.topk(logits, k, dim=1, largest=True, sorted=True)
    # print(f'topk_ind.shape: {topk_ind.shape}')
    # print(topk_ind)
    return topk_ind.tolist()


def find_most_overlapped_topk(labels_list, target, k=1):
    comb_ind_list = list(powerset(range(len(labels_list))))
    # print(len(comb_ind_list), comb_ind_list)
    labels_list = [set(x[1:]) if x[0]==target else set() for x in labels_list]
    common_labels_dict = {}
    for comb_ind in comb_ind_list:
        t_set = labels_list[comb_ind[0]]
        for i in comb_ind[1:]:
            t_set = t_set.intersection(labels_list[i])
        if len(t_set) > 0:
            common_labels_dict[comb_ind] = t_set
    # print(common_labels_dict)
    for comb_ind in sorted(common_labels_dict.keys(), key=lambda x: len(x), reverse=True):
        if len(common_labels_dict[comb_ind]) >= k:
            return comb_ind
    # print('decrease k to 1')
    k = 1
    for comb_ind in sorted(common_labels_dict.keys(), key=lambda x: len(x), reverse=True):
        if len(common_labels_dict[comb_ind]) >= k:
            return comb_ind

    # return all indices
    return list(range(len(labels_list)))


def my_select_ind(confs, target):
    labels_list = compute_topk_labels(confs, 5)
    comb_ind = find_most_overlapped_topk(labels_list, target, 2)
    confs = confs[comb_ind, target]
    ind = torch.argmax(confs).item()
    return comb_ind[ind]


def create_folder(folder):
    if os.path.exists(folder):
        assert os.path.isdir(folder), 'it exists but is not a folder'
    else:
        os.makedirs(folder)


class Tee(object):
    # from https://github.com/MKariya1998/GMI-Attack/blob/master/Celeba/utils.py
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if '...' not in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()


CONF_MASKS = torch.load('./conf_mask.pt')


def add_conf_to_tensor_(tensor, conf, color, highlight):
    """ Note: in-place modification on tensor
    """
    assert tensor.ndim == 3 and tensor.shape[0] == 3, 'tensor shape should be 3xHxW'
    mask = CONF_MASKS[conf]
    tensor[:, -46:-10, 10:120] = (1.-mask) * tensor[:, -46:-10, 10:120] + mask * color
    if highlight:
        width = 5
        tensor[0, :width, :] = 1.
        tensor[0, -width:, :] = 1.
        tensor[0, :, :width] = 1.
        tensor[0, :, -width:] = 1.


def add_conf_to_tensors(tensors, confs, color=torch.tensor([1., 0., 0.]).unsqueeze(1).unsqueeze(1), highlight_conf=None):
    """ Note: will clone the tensors to cpu
    """
    if len(tensors) != len(confs):
        raise AssertionError(f'{len(tensors)} != {len(confs)}, tensors.shape: {tensors.shape}')
    tensors = tensors.detach().cpu().clone()
    if highlight_conf is not None:
        highlight_confs = [x>=highlight_conf for x in confs]
    else:
        highlight_confs = [False] * len(confs)
    confs = [f'{x:.4f}' for x in confs]
    for i in range(len(tensors)):
        add_conf_to_tensor_(tensors[i], confs[i], color, highlight_confs[i])
    return tensors


def crop_and_resize(img, arch_name, resolution):
    img = crop_img(img, arch_name)
    img = resize_img(img, resolution)
    return img


if __name__ == '__main__':
    labels_list = [[1, 2377, 17, 1570, 2051],
                   [1, 2377, 17, 1570, 2051],
                   [1, 2377, 17, 2051, 1570],
                   [1, 2377, 17, 2051, 1570],
                   [1, 2377, 17, 1570, 2241],
                   [1, 848, 1915, 1806, 853],
                   [1, 1915, 853, 61, 1855],
                   [1, 35, 1915, 2217, 61]]
    labels_list = [[4, 1856, 2474, 674, 2171],
                   [4, 2235, 935, 2173, 844],
                   [4, 2611, 2173, 844, 935],
                   [4, 152, 27, 844, 2611],
                   [4, 1856, 199, 674, 2171],
                   [4, 2474, 2171, 1856, 139],
                   [4, 1027, 837, 10, 1440],
                   [4, 837, 1319, 1027, 1440]]
    target = 4
    k = 2
    comb_ind = find_most_overlapped_topk(labels_list, target, k)
    print('returned', comb_ind)
