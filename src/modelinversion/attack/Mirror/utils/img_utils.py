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
    # 'vgg16': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg_face_dag': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_5class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_8class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_9class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_16class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_24class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_10class_dp_sgd': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16_10class': [129.186279296875, 104.76238250732422, 93.59396362304688],
    'vgg16bn': [131.45376586914062, 103.98748016357422, 91.46234893798828],
    # 'resnet50': [131.0912, 103.8827, 91.4953],
    'resnet50_scratch_dag': [131.0912, 103.8827, 91.4953],
    'inception_resnetv1': [127.5, 127.5, 127.5],
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
    # 'vgg16': [1., 1., 1.],
    'vgg_face_dag': [1., 1., 1.],
    'vgg16_5class': [1., 1., 1.],
    'vgg16_8class': [1., 1., 1.],
    'vgg16_9class': [1., 1., 1.],
    'vgg16_16class': [1., 1., 1.],
    'vgg16_24class': [1., 1., 1.],
    'vgg16_10class_dp_sgd': [1., 1., 1.],
    'vgg16_10class': [1., 1., 1.],
    'vgg16bn': [1., 1., 1.],
    # 'resnet50': [1., 1., 1.],
    'resnet50_scratch_dag': [1., 1., 1.],
    'inception_resnetv1_vggface2': [128.0, 128.0, 128.0],
    'inception_resnetv1': [128.0, 128.0, 128.0],
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
    if arch_name not in ['resnet50_scratch_dag', 'vgg_face_dag'] and not arch_name.startswith('inception_resnetv1'):
        
        return image_tensor
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
    if arch_name not in ['resnet50_scratch_dag', 'vgg_face_dag'] and not arch_name.startswith('inception_resnetv1'):
        # print(f'>>>>>>>>>>>>> no normalize: {arch_name}')
        return image_tensor / 255
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

def get_input_resolution(arch_name):
    resolution = 224
    # to_grayscale = False
    # if arch_name.startswith('inception_resnetv1'):
    #     resolution = 160
    # elif arch_name == 'sphere20a':
    #     resolution = (112, 96)
    # elif arch_name.startswith('ccs19ami'):
    #     resolution = 64
    #     if 'rgb' not in arch_name:
    #         # to_grayscale = True
    #         pass
    # elif arch_name in ['azure', 'clarifai', ]:
    #     resolution = 256
    # elif arch_name == 'car_resnet34':
    #     resolution = 400
    
    if arch_name == 'vgg16':
        resolution = 64
    elif arch_name == 'ir152':
        resolution = 64
    elif arch_name == 'facenet64':
        resolution = 64
    elif arch_name == 'facenet':
        resolution = 112
    elif arch_name in ['resnet50_scratch_dag', 'vgg_face_dag']:
        resolution == 224
    elif arch_name.startswith('inception_resnetv1'):
        resolution == 160
    else:
        raise RuntimeError('arch name error')

    return resolution

def crop_and_resize(img, arch_name, resolution):
    img = crop_img(img, arch_name)
    img = resize_img(img, resolution)
    return img