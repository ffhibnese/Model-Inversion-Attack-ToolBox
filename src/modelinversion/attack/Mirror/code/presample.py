import torch
from torch import nn
from .genforce.get_genforce import get_genforce
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils.img_utils import *


def presample(result_dir, generator, sample_num=5000, batch_size=20, device='cuda'):

    z_dir = os.path.join(result_dir, 'z')
    w_dir = os.path.join(result_dir, 'w')
    img_dir = os.path.join(result_dir, 'img')

    os.makedirs(z_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    latent_dim = 512

    # z = torch.randn((sample_num, batch_size))

    # print(z_dir)

    with torch.no_grad():
        for i in tqdm(range(sample_num // batch_size)):
            z = torch.randn((batch_size, latent_dim)).to(device)
            w = generator.mapping(z)['w']
            torch.save(z, os.path.join(z_dir, f'sample_{i}.pt'))
            torch.save(w, os.path.join(w_dir, f'sample_{i}.pt'))
            fake = generator(w)
            torch.save(fake, os.path.join(img_dir, f'sample_{i}.pt'))


# def presample(result_dir, genforce_name, checkpoint_dir, sample_num = 5000, batch_size=20, device='cuda'):
#     generator, _ = get_genforce(genforce_name, device=device, checkpoint_dir=checkpoint_dir, use_w_space=True, use_z_plus_space=False, repeat_w=True, use_discri=False)

#     # result_dir = os.path.join(result_dir, genforce_name)

#     z_dir = os.path.join(result_dir, 'z')
#     w_dir = os.path.join(result_dir, 'w')
#     img_dir = os.path.join(result_dir, 'img')

#     os.makedirs(z_dir, exist_ok=True)
#     os.makedirs(w_dir, exist_ok=True)
#     os.makedirs(img_dir, exist_ok=True)

#     latent_dim = 512

#     # z = torch.randn((sample_num, batch_size))

#     # print(z_dir)

#     with torch.no_grad():
#         for i in tqdm(range(sample_num // batch_size)):
#             z = torch.randn((batch_size, latent_dim)).to(device)
#             w = generator.mapping(z)['w']
#             torch.save(z, os.path.join(z_dir, f'sample_{i}.pt'))
#             torch.save(w, os.path.join(w_dir, f'sample_{i}.pt'))
#             fake = generator(w)
#             torch.save(fake, os.path.join(img_dir, f'sample_{i}.pt'))
