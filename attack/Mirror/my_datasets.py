#!/usr/bin/env python3
# coding=utf-8
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

import celeba_partial256_dataset
from my_utils import crop_img, resize_img, denormalize, normalize, create_folder, Tee, crop_and_resize


class IndexedDataset(Dataset):
    """
    Wrapper to creat a dataset with elements denoted by indices
    """
    def __init__(self, dataset, indices):

        # currently only one-depth indexed dataset
        if isinstance(dataset, IndexedDataset):
            dataset = dataset.dataset
        assert not isinstance(dataset, IndexedDataset)
        self.dataset = dataset
        if isinstance(indices, torch.Tensor):
            assert len(indices.shape) == 1, 'Only 1D indices tensor is allowed'
            indices = indices.tolist()
        self.indices = indices
        assert isinstance(self.indices, list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.dataset[self.indices[idx]]
        elif isinstance(idx, list):
            indices = [self.indices[i] for i in idx]
            return IndexedDataset(self.dataset, indices)
        elif isinstance(idx, slice):
            indices = self.indices[idx]
            return IndexedDataset(self.dataset, indices)
        else:
            raise NotImplementedError(f'index type: {type(idx)}')

    def merge(self, dataset):
        assert isinstance(dataset, IndexedDataset)
        assert not isinstance(dataset.dataset, IndexedDataset)
        assert len(self.dataset) == len(dataset.dataset)  # a loose constraint to check dataset is same or not
        indices = self.indices[:]
        indices.extend(dataset.indices)
        indices.sort()
        return IndexedDataset(self.dataset, indices)

    def get_all_real_indices(self):
        return self.indices[:]


class CelebaLogitsDataSet(Dataset):
    """
    Dataset of tuple (img, logits, w) for celeba partial256
    """
    def __init__(self, arch_name, resolution, start_index=0, end_index=None, preload=False):
        self.celeba_dataset = celeba_partial256_dataset.Celeba(start_index=start_index, end_index=end_index)
        self.img_files = self.celeba_dataset.img_files
        self.arch_name = arch_name
        self.resolution = resolution

        self.pred_file_dir = os.path.join('./blackbox_attack_data/celeba_partial256', arch_name)

        self.preload = preload

        assert len(self.celeba_dataset) > 0

    def __len__(self):
        return len(self.celeba_dataset)

    def __getitem__(self, index):
        """
        Note: the image returned is [0, 1.]
        """
        w = 0  # fake w
        img, _, img_file = self.celeba_dataset[index]
        pred_file = f'{img_file}_logits.pt'  # 000001.jpg_logits.pt
        pred = torch.load(os.path.join(self.pred_file_dir, pred_file))

        img = crop_img(img, self.arch_name)
        img = resize_img(img, self.resolution)

        assert len(img.shape) == 3

        return img, pred, w

    def crop_and_resize(self, img):
        # img = crop_img(img, self.arch_name)
        # img = resize_img(img, self.resolution)
        return crop_and_resize(img, self.arch_name, self.resolution)


class StyleGANSampleDataSet(Dataset):
    """
    Dataset of pair (img, logits, w) pre-sampled for StyleGAN
    """
    def __init__(self, all_ws_pt_file, img_dir, arch_name, resolution, start_index=0, end_index=None, preload=False, target=None, use_dropout=False):
        self.all_ws = torch.load(all_ws_pt_file)
        self.img_dir = img_dir
        self.arch_name = arch_name
        self.resolution = resolution
        self.target = target  # used for azure and clarifai data only

        self.img_files = sorted(glob.glob(os.path.join(img_dir, 'smaple_*_img.pt0*')))
        if len(self.img_files) == 0:
            self.img_files = sorted(glob.glob(os.path.join(img_dir, 'sample_*_img.pt0*')))

        if self.arch_name == 'clarifai':
            self.img_files = [x for x in self.img_files if not x.endswith('jpg')]

        if self.arch_name == 'azure' or self.arch_name == 'clarifai':
            # only 100_000 images for azure
            self.img_files = self.img_files[:100_000]
            self.all_ws = self.all_ws[:100_000]
            assert start_index < 100_000 and end_index <= 100_000

        self.pred_files = sorted(glob.glob(os.path.join('./blackbox_attack_data/stylegan',
                                                        arch_name,
                                                        'use_dropout' if use_dropout else 'no_dropout',
                                                        'sample_*_img_logits.pt0*')))
        self.preload = preload

        assert len(self.img_files) > 0
        if not (len(self.img_files) == len(self.all_ws) == len(self.pred_files)):
            print(len(self.img_files), len(self.all_ws), len(self.pred_files))
            raise AssertionError('wrong length')

        for p, i in zip(self.pred_files, self.img_files):
            p = os.path.basename(p)
            i = os.path.basename(i)
            assert p == i[:-len('.pt000')]+'_logits'+i[-len('.pt000'):]

        if end_index is None:
            end_index = len(self.all_ws)

        self.all_ws = self.all_ws[start_index:end_index]
        self.img_files = self.img_files[start_index:end_index]
        self.pred_files = self.pred_files[start_index:end_index]

        if self.preload:
            self.img_files = [torch.load(x) for x in self.img_files]
            self.pred_files = [torch.load(x) for x in self.pred_files]

    def __len__(self):
        return len(self.all_ws)

    def __getitem__(self, index):
        w = self.all_ws[index]
        if self.preload:
            img = self.img_files[index]
            pred = self.pred_files[index]
        else:
            img = torch.load(self.img_files[index])
            pred = torch.load(self.pred_files[index])
        img = crop_img(img, self.arch_name)
        img = resize_img(img, self.resolution)
        assert len(img.shape) == 3
        assert len(w.shape) == 1
        return img, pred, w

    def crop_and_resize(self, img):
        # img = crop_img(img, self.arch_name)
        # img = resize_img(img, self.resolution)
        return crop_and_resize(img, self.arch_name, self.resolution)
