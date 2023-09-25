#!/usr/bin/env python3
# coding=utf-8
from collections import defaultdict
import os
import glob
from PIL import Image
import numpy as np
import shutil

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_celeba_img_name_dict():
    filename = '/data/share/celeba/list_identity_celeba.txt'
    lines = []
    with open(filename) as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                lines.append(line)
    lines = lines[2:]
    res = {}
    for line in lines:
        img, name = line.split('   ')
        res[img] = name
    return res


class Celeba(Dataset):
    def __init__(self, identity_file='/data/share/celeba/identity_CelebA_partial_256x256.txt', img_root='/data/share/celeba/partial_256x256', start_index=0, end_index=None):
        self.img_root = img_root
        self.labels = []
        self.names = []
        self.img_files = []
        self.label_imgfiles_dict = defaultdict(list)
        self.label_name_dict = {}
        self.name_label_dict = {}

        img_name_dict = load_celeba_img_name_dict()

        with open(identity_file) as in_file:
            for line in in_file:
                line = line.strip().split()
                if line:
                    self.names.append(img_name_dict[line[0]])
                    self.img_files.append(line[0])
                    self.labels.append(int(line[1]))
                    self.label_imgfiles_dict[self.labels[-1]].append(self.img_files[-1])
                    self.label_name_dict[self.labels[-1]] = self.names[-1]
                    self.name_label_dict[self.names[-1]] = self.labels[-1]

        assert len(self.img_files) == len(self.labels) == 108892

        if end_index is None:
            end_index = len(self.labels)

        self.img_files = self.img_files[start_index:end_index]
        self.labels = self.labels[start_index:end_index]
        self.names = self.names[start_index:end_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Note: the returned image is in [0., 1.]
        """
        img_file = self.img_files[index]
        label = self.labels[index]

        img = Image.open(os.path.join(self.img_root, img_file))
        img = np.array(img, dtype=np.uint8)
        img = img.transpose(2, 0, 1)
        assert len(img.shape) == 3  # assume color images and no alpha channel
        img = torch.from_numpy(img).float()/255.

        return img, label, img_file

    def print_info(self):
        print(f'# total images: {len(self.labels)}')
        print(f'# unique lables: {len(self.label_imgfiles_dict)}')
        label_imgfiles_list = [(k, v) for k, v in sorted(self.label_imgfiles_dict.items(), key=lambda x: len(x[1]), reverse=True)]
        for label, imgfiles in label_imgfiles_list[:10]:
            print(f'{label}: {len(imgfiles)}')
        print('.....')
        for label, imgfiles in label_imgfiles_list[-10:]:
            print(f'{label}: {len(imgfiles)}')

        print('#len>=30:', len(list(filter(lambda x: len(x[1])>=30, label_imgfiles_list))))

    def get_imgfiles_of_label(self, label):
        return [os.path.join(self.img_root, imgfile) for imgfile in self.label_imgfiles_dict[label]]


if __name__ == '__main__':
    celeba_dataset = Celeba()
    celeba_dataset.print_info()
    celeba_dataloader = DataLoader(celeba_dataset, batch_size=10, shuffle=False)
    for imgs, labels, img_files in celeba_dataloader:
        print(imgs.shape)
        print(labels.shape)
        print(img_files)
        break
