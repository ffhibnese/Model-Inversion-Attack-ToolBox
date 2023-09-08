# python3.7
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


def parse_args(model_name, num, batch_size, trunc_psi=0.7, trunc_layers=8):
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--num', type=int, default=num,
                        help='Number of samples to synthesize. '
                             '(default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=trunc_psi,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=trunc_layers,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    # return parser.parse_args([model_name, f'--num={num}', f'--batch_size={batch_size}', ])
    return parser.parse_args([model_name, ])


def main(device, model_name, num, batch_size, use_w_space=True, use_discri=True, repeat_w=True, use_z_plus_space=False, trunc_psi=0.7, trunc_layers=8):
    """Main function."""
    args = parse_args(model_name, num, batch_size, trunc_psi, trunc_layers)
    print(args)
    if args.num <= 0:
        return

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in '
                         f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'synthesis')
    os.makedirs(work_dir, exist_ok=True)

    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    if model_name.startswith('stylegan'):
        generator = build_generator(**model_config, repeat_w=repeat_w)
    else:
        generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print('Finish building generator.')

    # Build discriminator
    if use_discri:
        print(f'Building discriminator for model `{args.model_name}` ...')
        discriminator = build_discriminator(**model_config)
        print('Finish building discriminator.')
    else:
        discriminator = None

    # Load pre-trained weights.
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print('  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def fake_generator(code):
        # Sample and synthesize.
        # print(f'Synthesizing {args.num} samples ...')
        # code = torch.randn(args.batch_size, generator.z_space_dim).cuda()
        if use_z_plus_space:
            code = generator.mapping(code)['w']
            code = code.view(args.batch_size, generator.num_layers, generator.w_space_dim)
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

    def zero_grad(self):
        self.G.zero_grad()


if __name__ == '__main__':
    # main('stylegan_ffhq1024', 7, 7)
    # main('stylegan_ffhq256', 35, 35)
    main('stylegan_celeba_partial256', 35, 35)
