#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import glob
import math

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

use_w_space = False
repeat_w = True  # if False, opt w+ space
num_layers = 14  # 14 for stylegan w+ space with stylegan_celeba_partial256
# num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024
use_z_plus_space = False  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
trunc_psi = 0.7
trunc_layers = 8

# genforce_model = 'pggan_celebahq1024'
genforce_model = 'stylegan_celeba_partial256'
# genforce_model = 'stylegan_celebahq1024'
# genforce_model = 'stylegan2_ffhq1024'
# genforce_model = 'stylegan_ffhq256'
# genforce_model = 'stylegan_cat256'
# genforce_model = 'stylegan_animeportrait512'
# genforce_model = 'stylegan_animeface512'
# genforce_model = 'stylegan_artface512'
# genforce_model = 'stylegan_car512'
# genforce_model = 'stylegan2_car512'


if use_z_plus_space:
    use_w_space = True
    repeat_w = False
else:
    use_w_space = False
    repeat_w = True


def get_generator(batch_size, device):
    from genforce import my_get_GD
    # global use_w_space
    # if genforce_model.startswith('stylegan'):
    #     use_w_space = False
    use_discri = False
    generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space, trunc_psi=trunc_psi, trunc_layers=trunc_layers)
    return generator


@torch.no_grad()
def sample():
    device = 'cuda'
    latent_dim = 512
    batch_size = 100
    generator = get_generator(batch_size, device)
    RESOLUTION = 256

    iter_times = 1000 * (100 // batch_size)

    for i in tqdm(range(1, iter_times+1)):
        if use_z_plus_space:
            signal_file = './my_sample_zplus_w_space.signal'
        else:
            signal_file = './my_sample_z_w_space.signal'
        if not os.path.isfile(signal_file):
            with open(signal_file, 'w') as out_file:
                out_file.write('0')

        with open(signal_file) as in_file:
            line = in_file.readline().strip()
            if line and int(line) == 1:
                print('Stop iteration now')
                break

        if use_z_plus_space:
            latent_in = torch.randn(batch_size*num_layers, latent_dim, device=device)
            dirname = f'./stylegan_sample_zplus_{genforce_model}_{trunc_psi}_{trunc_layers}'
            filename = f'{dirname}/sample_{i}'
        else:
            latent_in = torch.randn(batch_size, latent_dim, device=device)
            dirname = f'./stylegan_sample_z_{genforce_model}_{trunc_psi}_{trunc_layers}'
            filename = f'{dirname}/sample_{i}'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        img_gen = generator(latent_in)
        torch.save(img_gen, f'{filename}_img.pt')

        if use_z_plus_space:
            latent_in = latent_in.view(batch_size, num_layers, latent_dim)
        torch.save(latent_in, f'{filename}_latent.pt')

        img_gen = F.resize(img_gen, (RESOLUTION, RESOLUTION))
        save_image(img_gen, f'{filename}.png', nrow=10)

    # collect all_ws.pt file
    all_ws = []
    all_latent_files = sorted(glob.glob(f'./{dirname}/sample_*_latent.pt*'))
    for i in tqdm(range(0, len(all_latent_files), batch_size)):
        latent_files = all_latent_files[i:i+batch_size]
        latent_in = [torch.load(f) for f in latent_files]
        latent_in = torch.cat(latent_in, dim=0)
        w = generator.G.mapping(latent_in.to(device))['w']
        all_ws.append(w)

    all_ws = torch.cat(all_ws, dim=0).cpu()
    torch.save(all_ws, f'./{dirname}_all_ws.pt')


if __name__ == '__main__':
    sample()
