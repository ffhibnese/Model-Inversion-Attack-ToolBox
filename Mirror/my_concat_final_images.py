#!/usr/bin/env python3
# coding=utf-8
import os
import glob
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as F


def load_image_to_tensor(img_file):
    inputs = Image.open(img_file)
    inputs = np.array(F.resize(inputs, (224, 224)), dtype=np.uint8)
    assert len(inputs.shape) == 3
    inputs = inputs.transpose(2, 0, 1)
    inputs = torch.clamp((torch.from_numpy(inputs).float() / 255.), 0., 1.)
    inputs = inputs.unsqueeze(0)
    return inputs


def concat_final_images(root_dir, nrow=1):
    generated_image_files = sorted(glob.glob(os.path.join(root_dir, 'img_*.jpg')))
    "img_label00101_id007_iter8251"
    print('\n'.join(generated_image_files[:2]))

    all_img_files = {}
    for generated_image_file in generated_image_files:
        img_file_base_name = os.path.basename(generated_image_file)
        img_file_base_name = img_file_base_name.split('_')
        id_in_batch = int(img_file_base_name[2][2:])
        all_img_files[id_in_batch] = generated_image_file

    all_img_files = [load_image_to_tensor(all_img_files[k]) for k in sorted(all_img_files.keys())]
    all_img_files = torch.cat(all_img_files, dim=0)
    print(f'all_img_files.shape: {all_img_files.shape}')
    filename = os.path.join(root_dir, 'all_img.jpg')
    # filename = os.path.join(os.path.dirname(root_dir), 'all_img.jpg')
    assert not (os.path.isfile(filename) or os.path.isdir(filename))
    print(f'save to {filename}')
    save_image(all_img_files, filename, nrow=nrow)


if __name__ == '__main__':
    root_dir = sys.argv[1].rstrip('/')
    nrow = int(sys.argv[2])
    concat_final_images(root_dir, nrow=nrow)
