from __future__ import print_function

import argparse
import os
import random
import numpy as np
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas

from utils import get_item, euclidean_dist
import utils
import data
import nets
import train

from csv_logger import CSVLogger, plot_csv
import json
import ipdb
import sys
from utils import mkdir, gaussian_logp
from tqdm import tqdm

import yaml
import matplotlib.pylab as plt
import itertools
from model_utils import (
    instantiate_generator,
    instantiate_discriminator,
    load_cls_embed,
    load_cls_z_to_lsm,
)

# Eval
from fid import calculate_frechet_distance
from eval_pretrained_face_classifier import PretrainedInsightFaceClassifier
from experimental import AttackExperiment


def save_checkpoint(args, state):
    torch.save(state, args.ckpt)


def maybe_load_checkpoint(args):
    if args.resume and os.path.exists(args.ckpt):
        return torch.load(args.ckpt)


def run_kde(
    x_train,
    x_eval,
    device,
    logvar=None,
    B=100,
    l2=False,
    return_details=False,
    detach=False,
    is_print=True,
    agg='logmeanexp',
    topk=5,
):
    assert len(x_train.shape) == len(x_eval.shape) == 2
    if logvar is None:
        logvar = torch.log(torch.var(x_train)) * torch.ones_like(x_train)
    logp_te = torch.zeros(x_train.size(0), x_eval.size(0)).to(
        device
    )  # (N_train, N_eval)
    x_eval = x_eval.to(device)
    if is_print:
        pbar = tqdm(range(0, x_train.size(0), B), desc='run_kde')
    else:
        pbar = range(0, x_train.size(0), B)
    for i in pbar:
        m_i = x_train[i : i + B].unsqueeze(1).to(device)
        if isinstance(logvar, torch.Tensor):
            lv_i = logvar[i : i + B].unsqueeze(1).to(device)
        else:
            lv_i = logvar * torch.ones_like(m_i)
        if l2:
            logp_te[i : i + B, :] = (
                -torch.pow(m_i - x_eval.unsqueeze(0), 2).sum([-1]).view(m_i.size(0), -1)
            )
        else:
            logp_te[i : i + B, :] = (
                gaussian_logp(m_i, 0.5 * lv_i, x_eval.unsqueeze(0), detach=detach)
                .sum([-1])
                .view(m_i.size(0), -1)
            )

    if agg == 'topk':
        logp_e = torch.topk(logp_te, k=topk, dim=0)[0].mean(0) - np.log(logp_te.size(0))
    elif agg == 'max':
        logp_e = torch.max(logp_te, dim=0)[0] - np.log(logp_te.size(0))
    elif agg == 'logmeanexp':
        logp_e = torch.logsumexp(logp_te, dim=0) - np.log(logp_te.size(0))
    else:
        raise ValueError

    if is_print:
        print(
            f"{torch.mean(logp_e)}, {torch.std(logp_e)}, {torch.max(logp_e)}, {torch.min(logp_e)}"
        )
    if not return_details:
        return logp_e
    else:
        return {'logp_e': logp_e, 'logp_te': logp_te}


def generate_N(inds, N, args, generator):
    fakes = []
    ys = []
    B = args.batchSize
    for _ in tqdm(range(N // B + 1), desc='generate_N'):
        with torch.no_grad():
            z = torch.randn(B, args.nz, 1, 1, device=device)
            yind = torch.randint(len(inds), (B,))
            y = inds[yind].to(device)
            # print(y)
            ys.append(y)
            fake = generator(z, y).detach()
            fakes.append(fake)
    fakes = torch.cat(fakes)[:N]
    ys = torch.cat(ys)[:N]
    return fakes, ys


def main(args):
    # Backward compat

    # Debug (i.e. quick) settings
    if args.db:
        pass

    # Data
    if not args.dummy_data:
        experiment = AttackExperiment(args.exp_config, device, args.db, fixed_id=-1)
        dat = experiment.dat
        args.imageSize = experiment.config['data']['image_size']
    else:
        config = yaml.load(open(f'configs/{args.exp_config}', 'r'))
        image_size = config['data']['image_size']
        nc = 1 if config['data']['name'] in ['mnist', 'chestxray'] else 3
        dat = {
            'X_train': torch.randn(1000, nc, image_size, image_size).to(device),
            'X_test': None,
            'Y_train': None,
            'Y_test': None,
            'nc': nc,
        }
        args.imageSize = config['data']['image_size']

    args.nc = dat['nc']
    utils.save_args(args, os.path.join(args.output_dir, f'args.json'))

    if args.model in ['l2_aux', 'dcgan_aux', 'kplus1gan']:
        target_extract_feat = experiment.target_extract_feat
        target_logsoftmax = experiment.target_logsoftmax
        target_logits = experiment.target_logits
        target_extract = {
            'embed': target_extract_feat,
            'logits': target_logits,
            'sm': lambda x: target_logsoftmax(x).exp(),
        }[args.context_type]
        nclass = (
            len(torch.unique(experiment.target_dataset['Y_train']))
            if 'nclass' not in experiment.target_dataset
            else experiment.target_dataset['nclass']
        )
        args.cdim = {'embed': experiment.cdim, 'logits': nclass, 'sm': nclass}[
            args.context_type
        ]
    else:
        args.cdim = 1  # dummy
        target_extract = None

    vutils.save_image(
        dat['X_train'][:100],
        '%s/data-train.jpeg' % (args.output_dir),
        normalize=True,
        nrow=10,
    )
    vutils.save_image(
        dat['X_train'][-100:],
        '%s/data-train1.jpeg' % (args.output_dir),
        normalize=True,
        nrow=10,
    )
    if dat['X_test'] is not None:
        vutils.save_image(
            dat['X_test'][:100],
            '%s/data-test.jpeg' % (args.output_dir),
            normalize=True,
            nrow=10,
        )
        vutils.save_image(
            dat['X_test'][:50],
            '%s/data-test5.jpeg' % (args.output_dir),
            normalize=True,
            nrow=5,
        )

    generator = instantiate_generator(args, device)
    print(generator)
    discriminator = instantiate_discriminator(args, dat['Y_train'], device)
    print(discriminator)

    # Initialize weights
    utils.weights_init(generator, args.g_init)
    utils.weights_init(discriminator, args.d_init)

    # Optim
    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=args.lrD,
        betas=(args.beta1, 0.999),
        weight_decay=args.wd,
    )
    optimizerG = optim.Adam(
        generator.parameters(),
        lr=args.lrD2lrG * args.lrD,
        betas=(args.beta1, 0.999),
        weight_decay=args.wd,
    )
    print('\nTraining with the following settings: {}'.format(args))

    # Evaluation setup
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)

    def evaluate(epoch, viz=True):
        epoch_log_dict = {'global_iteration': epoch}
        # Viz
        if generator.is_conditional:
            if args.model in ['l2_aux', 'dcgan_aux']:
                real_x = dat['X_train'][:100].to(device)

                # Generate
                with torch.no_grad():
                    c = target_extract(real_x / 2 + 0.5).detach()
                    fake = generator(fixed_noise[:100], c)
            else:
                fake_y = torch.from_numpy(np.arange(10).repeat(10)).to(device)
                fake = generator(fixed_noise[:100], fake_y).detach()
        else:
            fake = generator(fixed_noise).detach()
        vutils.save_image(
            fake[:100],
            '%s/viz_sample/sample_e%03d.jpeg' % (args.output_dir, epoch),
            normalize=True,
            nrow=10,
        )
        return epoch_log_dict

    # Log configs if training
    if not args.eval_only:
        #
        args.print('{} Generator: {}'.format(args.model.upper(), generator))
        args.print('{} Discriminator: {}'.format(args.model.upper(), discriminator))

        # Trace logging
        iteration_fieldnames = [
            'global_iteration',
            'd_loss',
            'real_acc',
            'fake_acc',
            'acc',
        ]
        if args.model == 'kplus1gan' and generator.is_conditional:
            iteration_fieldnames += ['class_acc']
        if args.model == 'kplus1gan':
            iteration_fieldnames += ['loss_distill']
        iteration_logger = CSVLogger(
            every=args.log_iter_every,
            fieldnames=iteration_fieldnames,
            filename=os.path.join(args.output_dir, 'iteration_log.csv'),
            resume=args.resume,
        )

        epoch_fieldnames = ['global_iteration']
        epoch_logger = CSVLogger(
            every=args.log_epoch_every,
            fieldnames=epoch_fieldnames,
            filename=os.path.join(args.output_dir, 'epoch_log.csv'),
            resume=args.resume,
        )
    else:
        # Evaluate saved models
        generator.load_state_dict(torch.load(os.path.join(args.ckpt_path)))
        epoch_log_dict = evaluate(-1, True)
        print('=' * 30 + "DONE" + '=' * 30)
        sys.exit(0)

    # Check for ckpt
    ckpt = maybe_load_checkpoint(args)
    if ckpt is not None:
        args.print("*" * 80 + "\nLoading ckpt \n" + "*" * 80)
        #
        start_epoch = ckpt['epoch']
        optimizerG.load_state_dict(ckpt['optimizerG'])
        optimizerD.load_state_dict(ckpt['optimizerD'])
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
    else:
        start_epoch = 0

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        args.print('*' * 100)
        args.print('Beginning of epoch {}'.format(epoch))
        args.print('*' * 100)

        # Maybe adjust topk_k
        if args.use_topk:
            gamma = max(args.topk_gamma**epoch, args.topk_min_gamma)
            args.topk_k = int(args.batchSize * gamma)
            print(epoch, args.topk_k)

        # Eval
        if epoch % args.eval_every == 0:
            epoch_log_dict = evaluate(epoch, args.viz_details)
            epoch_logger.writerow(epoch_log_dict)
            if len(epoch_log_dict) > 1:
                plot_csv(
                    epoch_logger.filename,
                    os.path.join(args.output_dir, 'epoch_plots.jpeg'),
                )

        # Ckpt
        state = {
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(args, state)
        # Save Models
        torch.save(
            generator.state_dict(), os.path.join(args.output_dir, 'generator.pt')
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(args.output_dir, 'discriminator.pt'),
        )
        if epoch in [20, 50, 100, 200, 300, 500, 1000]:
            torch.save(
                generator.state_dict(),
                os.path.join(args.output_dir, f'generator_{epoch}.pt'),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(args.output_dir, f'discriminator_{epoch}.pt'),
            )

        if args.model == 'dcgan':
            train.dcgan(
                dat['X_train'],
                generator,
                discriminator,
                optimizerG,
                optimizerD,
                args,
                epoch,
                iteration_logger,
                dat['Y_train'],
                target_extract,
            )
        elif args.model == 'kplus1gan':
            train.kplus1gan(
                dat['X_train'],
                generator,
                discriminator,
                optimizerG,
                optimizerD,
                args,
                epoch,
                iteration_logger,
                dat['Y_train'],
                target_extract,
                target_logsoftmax,
            )
        elif args.model in ['dcgan_aux', 'l2_aux']:
            # Precompute the context and cache them
            print('Precomputing the contexts')
            real_c = []
            with torch.no_grad():
                for i in range(0, len(dat['X_train']), args.batchSize):
                    stop = min(args.batchSize, len(dat['X_train'][i:]))
                    real_x = dat['X_train'][i : i + stop].to(device)
                    c = target_extract(real_x / 2 + 0.5).detach()
                    real_c.append(c.cpu())
            real_c = torch.cat(real_c)

            if args.model == 'dcgan_aux':
                train.dcgan_aux(
                    dat['X_train'],
                    generator,
                    discriminator,
                    optimizerG,
                    optimizerD,
                    args,
                    epoch,
                    iteration_logger,
                    dat['Y_train'],
                    target_extract,
                    real_c,
                )
            elif args.model == 'l2_aux':
                train.l2_aux(
                    dat['X_train'],
                    generator,
                    discriminator,
                    optimizerG,
                    optimizerD,
                    args,
                    epoch,
                    iteration_logger,
                    dat['Y_train'],
                    target_extract,
                    real_c,
                )
        elif args.model == 'mm':
            train.mm(
                dat['X_train'],
                generator,
                discriminator,
                optimizerG,
                optimizerD,
                args,
                epoch,
                iteration_logger,
            )
        else:
            raise ValueError(f"unknown option --model:{args.model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', type=str, default='')
    parser.add_argument(
        '--context_type', type=str, default='embed', choices=['embed', 'logits', 'sm']
    )
    # Data arguments
    parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2
    )
    parser.add_argument('--Ntrain', type=int, default=60000, help='training set size')
    parser.add_argument('--Ntest', type=int, default=10000, help='test set size ')
    parser.add_argument('--dataset_size', type=int, default=-1)

    # Model arguments
    parser.add_argument('--model', required=True, help=' dcgan | mm')
    parser.add_argument('--use_labels', required=True, type=int)
    parser.add_argument(
        '--nz', type=int, default=100, help='size of the latent z vector'
    )
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--g_init', type=str, default='N02')
    parser.add_argument('--d_init', type=str, default='N02')
    parser.add_argument('--g_sn', type=int, default=0)
    parser.add_argument('--g_z_scale', type=float, default=1)
    parser.add_argument(
        '--g_conditioning_method', type=str, default='mul', choices=['add', 'mul']
    )
    parser.add_argument('--g_norm', type=str, default='bn', choices=['bn', 'in'])

    # Optimization arguments
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=1000, help='number of epochs to train for'
    )
    parser.add_argument(
        '--lrD', type=float, default=0.0002, help='learning rate, default=0.0002'
    )
    parser.add_argument(
        '--lrD2lrG', type=float, default=1, help='learning rate, default=0.0002'
    )
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--wd', type=float, default=0.0, help='wd for adam')
    parser.add_argument('--seed', type=int, default=2019, help='manual seed')
    parser.add_argument('--d_noise', type=float, default=0)
    parser.add_argument(
        '--augment',
        nargs='?',
        const='',
        type=str,
        default='',
        help='see DiffAugment_pytorch.py',
    )

    # Checkpointing and Logging arguments
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--log_iter_every', type=int, default=100)
    parser.add_argument('--log_epoch_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument(
        '--save_ckpt_every', type=int, default=100, help='when to save checkpoint'
    )
    parser.add_argument(
        '--save_imgs_every', type=int, default=1, help='when to save generated images'
    )
    parser.add_argument(
        '--num_gen_images',
        type=int,
        default=150,
        help='number of images to generate for inspection',
    )
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--resume_from_local_ckpt', type=int, default=0)
    parser.add_argument('--user', type=str, default='wangkuan')
    parser.add_argument('--eval_size', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--viz_details', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='')

    # Discriminator Arch
    parser.add_argument(
        '--disc_config',
        type=str,
        default='disc0.yaml',
        help='look in dir ./disc_config/*',
    )
    parser.add_argument(
        '--disc_kwargs',
        type=str,
        default='',
        help='convenience kwargs, format(<name:type.value>,) type={i,f,s}',
    )
    parser.add_argument('--n_conditions', type=int, default=1)

    # Generator Arch
    parser.add_argument('--gen', type=str, default='basic', help='basic | conditional')

    # MemScore

    # Inverting auxiliary dataset
    # parser.add_argument('--target_dataset',nargs='?', const='',  type=str, default='')
    parser.add_argument('--l2_aux_reg', type=float, default=0)
    # parser.add_argument('--cls_path', type=str, default='')

    # Baseline methods
    parser.add_argument('--kplus1_distill_lambda', type=float, default=0)
    parser.add_argument('--lambda_diversity', type=float, default=0)

    # Topk training
    parser.add_argument('--use_topk', type=int, default=0)
    parser.add_argument('--topk_gamma', type=float, default=0.99)
    parser.add_argument('--topk_min_gamma', type=float, default=0.75)

    # Dev
    parser.add_argument('--db', type=int, default=0)
    parser.add_argument('--dummy_data', type=int, default=0)
    parser.add_argument('--eval_only', type=int, default=0)
    args = parser.parse_args()

    # Discs
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir, 'sample_pt'))
    mkdir(os.path.join(args.output_dir, 'viz_sample'))
    mkdir(os.path.join(args.output_dir, 'viz_inferece'))
    mkdir(os.path.join(args.output_dir, 'viz_mm_sample'))
    mkdir(os.path.join(args.output_dir, 'viz_memscore_stats'))
    args.jobid = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else -1
    utils.save_args(args, os.path.join(args.output_dir, f'args.json'))

    # Global Config
    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Logs
    log_file = os.path.join(args.output_dir, 'log.txt')
    log = open(log_file, 'w')

    def myprint(*content):
        print(*content)
        print(*content, file=log)
        log.flush()

    args.print = myprint
    args.print(f"Slurm ID: {args.jobid}")
    args.ckpt = f"/checkpoint/{args.user}/{os.environ['SLURM_JOB_ID']}/ckpt.pt"
    #
    if args.resume_from_local_ckpt:
        args.ckpt = os.path.join(args.output_dir, 'ckpt.pt')

    main(args)
