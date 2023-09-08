#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import torchvision.models as models

import net_sphere

from my_utils import create_folder, Tee
from my_whitebox_mirror_helper import mirror_attack
from my_target_models import get_model, get_input_resolution


random.seed(0)


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device = device

    args.final_img_dirname = f'final_images/{args.exp_name}'
    args.tmp_img_dirname = f'generations/{args.exp_name}'

    create_folder(args.tmp_img_dirname)
    Tee(os.path.join(args.tmp_img_dirname, 'output.log'), 'w')

    create_folder(f'{args.tmp_img_dirname}/images/')
    create_folder(args.final_img_dirname)

    net = get_model(args.arch_name, device, args.use_dropout)

    verifier_device = torch.device(f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    net_verifier = get_model(args.arch_name, verifier_device, args.use_dropout)

    print(args)

    args.image_resolution = get_input_resolution(args.arch_name)

    if args.arch_name == 'sphere20a':
        args.criterion = net_sphere.MyAngleLoss()
    else:
        args.criterion = nn.CrossEntropyLoss()

    args.targets = list(map(int, args.target.split(',')))
    print(args.targets)

    mirror_attack(args, net, net_verifier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', default=20000, type=int, help='optimization epochs')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--loss_class_ce', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--target', help='the only one target to invert, or multiple targets separated by ,')
    parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std; set 0. to unbound')
    parser.add_argument('--trunc_psi', type=float, default=0.7, help='truncation percentage')
    parser.add_argument('--trunc_layers', type=int, default=8, help='num of layers to truncate')
    parser.add_argument('--genforce_model', choices=['pggan_celebahq1024', 'stylegan_celeba_partial256', 'stylegan_ffhq256', 'stylegan2_ffhq1024', 'stylegan_cat256', 'stylegan_animeportrait512', 'stylegan_animeface512', 'stylegan_artface512', 'stylegan_car512', ], default='stylegan_celeba_partial256', help='genforce gan model')
    parser.add_argument('--pre_samples_dir', default='./stylegan_sample_z_stylegan_celeba_partial256_0.7_8', help='pre-generated samples of gan')
    parser.add_argument('--all_ws_pt_file', default='./stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt', help='all ws pt file')
    parser.add_argument('--latent_space', default='w', choices=['w', 'z', 'w+', 'z+'], help='evaluate batch with another model')
    parser.add_argument('--use_w_mean', action='store_true', help='start optimizing with w_mean')
    parser.add_argument('--to_truncate_z', action='store_true', help='truncate z vectors')
    parser.add_argument('--z_std_ce', type=float, default=1., help='set the bound for z space bound mean+-x*std')
    parser.add_argument('--loss_discri_ce', type=float, default=0., help='coefficient for discri loss')
    parser.add_argument('--naive_clip_w_bound', default=0., type=float, help='use naive clip in w')
    parser.add_argument('--energy', action='store_true', help='use energy term')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('--loss_l2_bound_latent_ce', type=float, default=0., help='ce to bound l2 distance between the optimized latent vectors and the starting vectors.')
    parser.add_argument('--save_every', type=int, default=100, help='how often to save the intermediate results')
    parser.add_argument('--use_cache', action='store_true')

    args = parser.parse_args()
    # print(args)

    if args.genforce_model is None:
        args.latent_space = None
    else:
        if args.genforce_model.startswith('stylegan'):
            assert args.latent_space is not None
        else:
            # other latent spaces are only meaningful for stylegan*
            args.latent_space = 'z'

    if args.genforce_model != 'stylegan2_ffhq1024':
        torch.backends.cudnn.benchmark = True

    run(args)


if __name__ == '__main__':
    main()
