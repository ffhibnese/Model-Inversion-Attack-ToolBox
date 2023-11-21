
"""This file is modifed from synthesize.py. The goal is to return a generator which output an image in range [0., 1.]"""

import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torchvision.utils import save_image

from .models import MODEL_ZOO
from .models import build_generator, build_discriminator
from .utils.misc import bool_parser
from .utils.visualizer import HtmlPageVisualizer

def postprocess(images):
    """change the range from [-1, 1] to [0., 1.]"""
    images = torch.clamp((images + 1.) / 2., 0., 1.)
    return images

def get_genforce(model_name, device, checkpoint_dir, use_discri=True, use_w_space=True, use_z_plus_space=False, repeat_w=True):
    
    trunc_psi = 0.7
    trunc_layers = 8
    
    if model_name not in MODEL_ZOO:
        raise RuntimeError(f'model name `{model_name}` is not in model zoo')
    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')
    
    print(f'Building generator for model `{model_name}`')
    if model_name.startswith('stylegan'):
        generator = build_generator(**model_config, repeat_w=repeat_w)
    else:
        generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=trunc_psi,
                            trunc_layers=trunc_layers)
    
    # Build discriminator
    if use_discri:
        print(f'Building discriminator for model `{model_name}` ...')
        discriminator = build_discriminator(**model_config)
    else:
        discriminator = None
        
    # load checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, 'genforce', f'{model_name}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f'Download checkpoint {model_name} from {url} ...')
        subprocess.call(['wget', '--quiet', '-O', ckpt_path, url])
        
    checkpoint = torch.load(ckpt_path)
    
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.to(device)
    generator.eval()
    if use_discri:
        discriminator.load_state_dict(checkpoint['discriminator'])
        discriminator = discriminator.to(device)
        discriminator.eval()
    print('Finish loading checkpoint.')
    
    def fake_generator(code):
        # Sample and synthesize.
        # print(f'Synthesizing {args.num} samples ...')
        # code = torch.randn(args.batch_size, generator.z_space_dim).cuda()
        if use_z_plus_space:
            code = generator.mapping(code)['w']
            code = code.view(-1, generator.num_layers, generator.w_space_dim)
        images = generator(code, **synthesis_kwargs, use_w_space=use_w_space)['image']
        images = postprocess(images)
        # save_image(images, os.path.join(work_dir, 'tmp.png'), nrow=5)
        # print(f'Finish synthesizing {args.num} samples.')
        return images

    return Fake_G(generator, fake_generator), discriminator

class Fake_G:

    def __init__(self, G, g_function):
        self.G = G
        self.g_function = g_function

    def __call__(self, code):
        # print(f'code.shape {code.shape}')
        return self.g_function(code)
    
    def mapping(self, code, label=None):
        return self.G.mapping(code, label=None)

    def zero_grad(self):
        self.G.zero_grad()