'''
    Source: https://github.com/mseitzer/pytorch-fid
    Modified code to be compatible with our attack pipeline

    Copyright [2021] [Maximilian Seitzer]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

'''

import numpy as np
import pytorch_fid.fid_score
import torch
import torchvision.transforms.functional as F
from pytorch_fid.inception import InceptionV3

IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp')


class FID_Score:
    def __init__(self, device, crop_size=None, batch_size=128, dims=2048, num_workers=8, gpu_devices=[]):
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.crop_size = crop_size
        self.pred_arr_fake = []
        self.pred_arr_gt = []

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx]).to(self.device)
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(
                inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(device)

    def set(self, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2

    # 获取对应层下的fid
    def get_fid(self):
        pred_arr_gt = np.concatenate(self.pred_arr_gt, axis=0)
        mu1 = np.mean(pred_arr_gt, axis=0)
        sigma1 = np.cov(pred_arr_gt, rowvar=False)

        pred_arr_fake = np.concatenate(self.pred_arr_fake, axis=0)
        mu2 = np.mean(pred_arr_fake, axis=0)
        sigma2 = np.cov(pred_arr_fake, rowvar=False)
        fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
            mu1, sigma1, mu2, sigma2)
        return fid_value

    def compute_fid(self):
        self.compute_statistics(self.dataset_1)
        self.compute_statistics(self.dataset_2, fake=True)
        # m1, s1 = self.compute_statistics(self.dataset_1, rtpt)
        # m2, s2 = self.compute_statistics(self.dataset_2, rtpt, True)
        # fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
        #     m1, s1, m2, s2)
        # return fid_value

    # 计算FID
    def compute_statistics(self, dataset, fake=False):
        self.inception_model.eval()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                # 表示该数据集是重建图像数据集
                if fake:
                    x = create_image(x, crop_size=self.crop_size, resize=299)
                x = x.to(self.device)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        if fake:
            self.pred_arr_fake.append(pred_arr)
        else:
            self.pred_arr_gt.append(pred_arr)
        # mu = np.mean(pred_arr, axis=0)
        # sigma = np.cov(pred_arr, rowvar=False)
        # return mu, sigma

# 对图像进行变换
def create_image(imgs, crop_size=None, resize=None):
    if crop_size is not None:
        imgs = F.center_crop(imgs, (crop_size, crop_size))
    if resize is not None:
        imgs = F.resize(imgs, resize, antialias=True)
    return imgs
