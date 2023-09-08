#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import glob
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as F

from my_utils import crop_img, resize_img, create_folder, add_conf_to_tensors
from my_test_confidence import test_final_result


@torch.no_grad()
def concat_final_images_from_latent(arch_name, dirs, args):

    batch_size = args.bs

    if arch_name == 'vgg16bn':
        test_arch_name = 'vgg16'
    elif arch_name == 'resnet50':
        test_arch_name = 'inception_resnetv1_vggface2'
    elif arch_name == 'vgg16':
        test_arch_name = 'vgg16bn'
    elif arch_name == 'inception_resnetv1_vggface2':
        test_arch_name = 'resnet50'
    elif arch_name == 'inception_resnetv1_casia':
        test_arch_name = 'sphere20a'
    elif arch_name == 'sphere20a':
        test_arch_name = 'inception_resnetv1_casia'
    elif arch_name == 'cat_resnet18':
        test_arch_name = None
    elif arch_name == 'car_resnet34':
        test_arch_name = None
    else:
        raise AssertionError('wrong arch_name')

    assert check_dirs(dirs)

    latent_files = [os.path.join(d, 'latent_inputs.pt') for d in dirs]
    missing_latent_files = [x for x in latent_files if not os.path.isfile(x)]
    if missing_latent_files:
        print('files are missing!')
        print('\n'.join(missing_latent_files))
        exit()

    print('computing confs......')
    all_stylegan_images = None
    all_confs, correct_cnt, topk_correct_cnt, total_cnt, l2_dist, conf_diff_score, my_selected_inds = test_final_result(arch_name, dirs, external_args=args)

    all_images = []
    test_all_confs, test_correct_cnt, test_topk_correct_cnt, test_total_cnt, test_l2_dist, test_conf_diff_score, _ = test_final_result(test_arch_name, dirs, external_args=args)

    device = 'cuda'

    use_w_space = 'w' in args.latent_space
    repeat_w = '+' not in args.latent_space   # if False, opt w+ space
    # num_layers = 14  # 14 for stylegan w+ space
    # num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024

    genforce_model = args.genforce_model
    if not genforce_model.startswith('stylegan'):
        use_w_space = False

    if args.latent_space == 'z+':
        use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
        use_w_space = True
    else:
        use_z_plus_space = False

    def get_generator(batch_size, device):
        from genforce import my_get_GD
        use_discri = False
        generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space)
        return generator

    if args.my_select:
        batch_size = 1

    generator = get_generator(batch_size, device)

    latent_files = [os.path.join(d, 'latent_inputs.pt') for d in dirs]

    all_images = []
    for i, latent_file in enumerate(latent_files):
        if all_stylegan_images is None:
            latent_in = torch.load(latent_file)
            if args.my_select:
                my_selected_ind = my_selected_inds[i]
                latent_in = latent_in[my_selected_ind:my_selected_ind+1]
            assert len(latent_in) == batch_size
            images = generator(latent_in)
        else:
            images = all_stylegan_images[i*batch_size:(i+1)*batch_size]
        # images = crop_img(images, arch_name)
        images = resize_img(images, 256)  # currently, just resize every generated images to 256
        if args.add_conf:
            if args.my_select:
                images = add_conf_to_tensors(images, all_confs[i])
            else:
                images = add_conf_to_tensors(images, all_confs[i])  # , highlight_conf=0.8)
        all_images.append(images)

    all_images = torch.cat(all_images, dim=0)

    if args.use_dropout:
        save_image(all_images, f'./tmp/all_images_{arch_name}_{args.genforce_model}_{args.latent_space}_{args.dir_prefix}_dropout.png', nrow=batch_size)
    else:
        save_image(all_images, f'./tmp/all_images_{arch_name}_{args.genforce_model}_{args.latent_space}_{args.dir_prefix}.png', nrow=batch_size)

    print(arch_name)
    print(f'top1 acc: {correct_cnt}/{total_cnt} = {correct_cnt/total_cnt:.4f}')
    print(f'topk acc: {topk_correct_cnt}/{total_cnt} = {topk_correct_cnt/total_cnt:.4f}')
    print(f'l2_dist: {l2_dist:.2f}')
    print(f'conf_diff_score: {conf_diff_score:.2f}')

    if test_arch_name is not None:
        print(test_arch_name)
        print(f'top1 acc: {test_correct_cnt}/{test_total_cnt} = {test_correct_cnt/test_total_cnt:.4f}')
        print(f'topk acc: {test_topk_correct_cnt}/{test_total_cnt} = {test_topk_correct_cnt/test_total_cnt:.4f}')
        print(f'l2_dist: {test_l2_dist:.2f}')
        print(f'conf_diff_score: {test_conf_diff_score:.2f}')


def check_dirs(dirs):
    good = True
    for d in dirs:
        if not os.path.isdir(d):
            print(f'{d} does not exist.')
            good = False
    return good


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_space', choices=['w', 'z', 'z+', 'w+'], help='evaluate batch with another model')
    parser.add_argument('--add_conf', action='store_true', help='add conf to images')
    parser.add_argument('--my_select', action='store_true', help='select the most consistent image')
    parser.add_argument('--my_ce_select', action='store_true', help='select the most confident image')
    parser.add_argument('--use_dropout', action='store_true', help='check the result with dropout strategy')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('genforce_model', choices=['pggan_celebahq1024', 'stylegan_celeba_partial256', 'stylegan_ffhq256', 'stylegan2_ffhq1024', 'stylegan_cat256', 'stylegan_artface512', 'stylegan_car512', ], help='genforce gan model')
    parser.add_argument('arch_name', choices=['vgg16', 'vgg16bn', 'resnet50', 'inception_resnetv1_vggface2', 'inception_resnetv1_casia', 'sphere20a', 'cat_resnet18', 'car_resnet34', ], help='test arch name')
    parser.add_argument('dir_prefix', help='the prefix of the dir names')
    parser.add_argument('targets', help='targets separated by,')

    args = parser.parse_args()
    print(args)

    arch_name = args.arch_name

    if not args.genforce_model.startswith('stylegan'):
        # other latent spaces are only meaningful for stylegan*
        args.latent_space = 'z'

    if '-' in args.targets:
        t0, t1 = args.targets.strip().split('-')
        targets = list(range(int(t0), int(t1)+1))
    else:
        targets = args.targets.strip().split(',')
    dirs = [f'./final_images/{args.dir_prefix}_{i}' for i in targets]

    concat_final_images_from_latent(arch_name, dirs, args)
