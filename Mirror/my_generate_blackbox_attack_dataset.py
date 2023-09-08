#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os
import random
import sys

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

import celeba_partial256_dataset
from my_utils import crop_img, resize_img, normalize, create_folder, Tee
from my_target_models import get_model, get_input_resolution


random.seed(0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='set the seed')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('dataset', choices=['stylegan', 'celeba_partial256'], help='use which dataset')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    exp_name = os.path.join('blackbox_attack_data', args.dataset, args.arch_name, 'use_dropout' if args.use_dropout else 'no_dropout')
    create_folder(exp_name)
    Tee(os.path.join(exp_name, 'output.log'), 'w')
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    net = get_model(args.arch_name, device, args.use_dropout)

    resolution = get_input_resolution(args.arch_name)

    arch_name = args.arch_name

    if args.dataset == 'stylegan':
        img_dir = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8/'
        imgs_files = sorted(glob.glob(os.path.join(img_dir, 'sample_*_img.pt')))

        assert len(imgs_files) > 0

        for img_gen_file in tqdm(imgs_files):
            save_filename = os.path.join(exp_name, os.path.basename(img_gen_file)[:-3]+'_logits.pt')
            fake = torch.load(img_gen_file).to(device)
            fake = crop_img(fake, arch_name)
            fake = normalize(resize_img(fake*255., resolution), args.arch_name)
            prediction = net(fake)
            if arch_name == 'sphere20a':
                prediction = prediction[0]
            torch.save(prediction, save_filename)
    elif args.dataset == 'celeba_partial256':
        kwargs = {'num_workers': 16, 'pin_memory': True} if not args.no_cuda else {}
        celeba_dataset = celeba_partial256_dataset.Celeba()
        celeba_dataloader = DataLoader(celeba_dataset, batch_size=100, shuffle=False, **kwargs)
        for imgs, labels, img_files in tqdm(celeba_dataloader):
            imgs = imgs.to(device)
            imgs = crop_img(imgs, arch_name)
            imgs = normalize(resize_img(imgs*255., resolution), args.arch_name)
            prediction = net(imgs)
            if arch_name == 'sphere20a':
                prediction = prediction[0]
            for i, img_file in enumerate(img_files):
                save_filename = os.path.join(exp_name, img_file+'_logits.pt')
                torch.save(prediction[i].clone().cpu(), save_filename)


if __name__ == '__main__':
    main()
