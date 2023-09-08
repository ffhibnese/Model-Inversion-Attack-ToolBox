from __future__ import print_function

import argparse
import os
import socket
import numpy as np
import torch
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import pandas
import utils
from csv_logger import CSVLogger, plot_csv
from utils import mkdir
from tqdm import tqdm
from experimental import AttackExperiment


def select_highest_pyx_given_x(x, target_logsoftmax, Nperclass, target_id=None):
    with torch.no_grad():
        logps = []
        for start in range(0, len(x), 1000):
            logp = target_logsoftmax(
                x[start:start + 1000].cuda() / 2 + 0.5)[:, args.fixed_id]
            logps.append(logp)
        logps = torch.cat(logps)
        best_idx = torch.sort(logps)[1].cpu()[-Nperclass:]
    return best_idx


def main(args):
    # db config
    if args.db:
        args.target_dataset = args.dataset = 'celeba-db'

    # backward compat
    args.nc = 3

    # Experiment setup
    experiment = AttackExperiment(args.exp_config, device, args.db)
    target_logsoftmax = experiment.target_logsoftmax
    target_eval_runner = experiment.target_eval_runner
    generator = experiment.generator
    discriminator = experiment.discriminator
    nclass = experiment.target_dataset['nclass']
    gan_method = experiment.gan_method

    # Prior Model
    if args.prior_model == 'disc':
        def prior_loss_func(x):
            return -discriminator(x)
    elif args.prior_model == '0':
        args.gan_gd_prior_lambda = 0

    # GD
    best_fakes = []
    for run in range(3):
        best_eval_acc = -1
        best_fake = None
        best_noise = None

        # Logging
        iteration_fieldnames = ['global_iteration', 'loss',
                                'train_target_acc', 'eval-acc', 'eval-top5_acc']
        mkdir(os.path.join(args.output_dir, f'run{run}'))
        iteration_logger = CSVLogger(every=args.log_iter_every,
                                     fieldnames=iteration_fieldnames,
                                     filename=os.path.join(args.output_dir, f'run{run}', 'iteration_log.csv'),
                                     resume=args.resume)

        # Init z
        if args.method == 'genmi':
            if args.init_rs:
                # Select noise that has the highest p(y|x) as init
                N_init = 10000
                noise0 = args.gan_gd_init_scale * \
                    torch.randn(N_init, experiment.gan_args.nz,
                                1, 1, device=device)

                with torch.no_grad():
                    logps = []
                    for start in range(0, N_init, args.batchSize):
                        z0 = noise0[start:start + args.batchSize]
                        if gan_method == 'dcgan_aux':
                            fake_y = torch.ones((args.batchSize,)) * args.fixed_id
                            fake_y_onehot = torch.eye(
                                nclass)[fake_y.long()].to(device)
                            fake = generator(z0, fake_y_onehot)
                        else:
                            fake = generator(z0)
                        logp = target_logsoftmax(fake / 2 + 0.5)[:, args.fixed_id]
                        logps.append(logp)
                    logps = torch.cat(logps)
                    best_idx = torch.sort(logps)[1][-args.batchSize:]
                noise = noise0[best_idx]

            else:
                noise = args.gan_gd_init_scale * \
                    torch.randn(args.batchSize, experiment.gan_args.nz,
                                1, 1, device=device)

            noise.requires_grad_()
            noise_optimizer = optim.SGD(
                [noise], lr=args.gan_gd_lr, momentum=args.gan_gd_m, weight_decay=args.gan_gd_wd, nesterov=False)
        elif args.method == 'gmi':
            C, H, W = experiment.target_dataset['X_train'][0].shape
            noise = -1 + 2 * torch.rand(args.batchSize, C, H, W, device=device)

            noise.requires_grad_()
            noise_optimizer = optim.SGD(
                [noise], lr=args.gan_gd_lr, momentum=args.gan_gd_m, weight_decay=args.gan_gd_wd, nesterov=False)
        else:
            raise ValueError()

        pbar = tqdm(range(0, args.gan_gd_steps), desc='Opt')
        for i in pbar:
            noise_optimizer.zero_grad()
            if args.method == 'genmi':
                if gan_method == 'dcgan_aux':
                    fake_y = torch.ones((len(noise),)) * args.fixed_id
                    fake_y_onehot = torch.eye(nclass)[fake_y.long()].to(device)
                    fake = generator(noise, fake_y_onehot)
                else:
                    fake = generator(noise)
            elif args.method == 'gmi':
                fake = noise.clamp(-1, 1)
            else:
                raise ValueError()

            # Compute loss
            lsm = target_logsoftmax(fake / 2 + .5)
            fake_y = args.fixed_id * \
                torch.ones(args.batchSize).to(device).long()
            target_loss = -lsm.gather(1, fake_y.view(-1, 1)).mean()
            train_target_acc = (lsm.max(1)[1] == fake_y).float().mean().item()

            loss = 0
            if args.gan_gd_lambda > 0:
                loss = loss + args.gan_gd_lambda * target_loss

            if args.gan_gd_prior_lambda > 0:
                loss_prior = prior_loss_func(fake).mean()
                loss = loss + args.gan_gd_prior_lambda * loss_prior

            loss.backward()
            noise_optimizer.step()

            pbar.set_postfix_str(s=f'Loss: {loss.item():.2f}, Acc: {train_target_acc:.3f}', refresh=True)
            if i % 100 == 0:
                vutils.save_image(fake[:64], '%s/viz_sample/sample_run%03d_i%03d.jpeg' %
                                  (args.output_dir, run, i), normalize=True, nrow=8)
                D = target_eval_runner.evaluate(
                    fake, args.fixed_id * torch.ones(len(fake)).to(device).long(), None)
                D['train-target-acc'] = train_target_acc
                pandas.Series(D).to_csv(os.path.join(args.output_dir, f'gan_gd_tdr_metrics_run{run}_i{i}.csv'))

                stats_dict = {
                    'global_iteration': i,
                    'loss': loss.item(),
                    'train_target_acc': train_target_acc
                }
                for field in iteration_fieldnames[3:]:
                    stats_dict[field] = D[field[5:]]
                iteration_logger.writerow(stats_dict)
                plot_csv(iteration_logger.filename, os.path.join(args.output_dir, f'run{run}', 'iteration_plots.jpeg'))

                # If best
                if stats_dict['eval-acc'] > best_eval_acc:
                    best_eval_acc = stats_dict['eval-acc']
                    best_fake = fake.detach().clone()
                    best_noise = noise.detach().clone()
                    torch.save(best_noise, os.path.join(args.output_dir, f'best_noise_run{run}.pt'))
                    torch.save(best_fake, os.path.join(args.output_dir, f'best_fake_run{run}.pt'))
                    with open(os.path.join(args.output_dir, f'best_iter_run{run}.txt'), 'w') as f:
                        f.write(f"{i}")

                torch.save(noise.detach().clone().cpu(), os.path.join(args.output_dir, f'last_noise_run{run}.pt'))
                torch.save(fake.detach().clone().cpu(), os.path.join(args.output_dir, f'last_fake_run{run}.pt'))

        best_fakes.append(best_fake)
    best_fakes = torch.cat(best_fakes)

    def _final(fakes, name):
        vutils.save_image(fakes[:64], f'{args.output_dir}/viz_sample/sample__{name}.jpeg', normalize=True, nrow=8)
        D = target_eval_runner.evaluate(
            fakes, args.fixed_id * torch.ones(len(fakes)).to(device).long(), None)
        pandas.Series(D).to_csv(os.path.join(args.output_dir, f'gan_gd_all_{name}.csv'))

    _final(best_fakes, 'before_rs')
    #  Select the most probable samples
    best_idx = select_highest_pyx_given_x(
        best_fakes, target_logsoftmax, 1, target_id=args.fixed_id)
    _final(best_fakes[best_idx].cuda(), 'after_rs_1')

    best_idx = select_highest_pyx_given_x(
        best_fakes, target_logsoftmax, 10, target_id=args.fixed_id)
    _final(best_fakes[best_idx].cuda(), 'after_rs_10')
    best_idx = select_highest_pyx_given_x(
        best_fakes, target_logsoftmax, 100, target_id=args.fixed_id)
    _final(best_fakes[best_idx].cuda(), 'after_rs_100')
    torch.save(best_fakes[best_idx], os.path.join(
        args.output_dir, "after_rs_100_samples.pt"))
    best_idx = select_highest_pyx_given_x(
        best_fakes, target_logsoftmax, 500, target_id=args.fixed_id)
    _final(best_fakes[best_idx].cuda(), 'after_rs_500')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='genmi', choices=['gmi', 'genmi'], help='gmi optimizes in x-space, genmi in z-space')
    # Data arguments
    parser.add_argument('--exp_config', type=str, required=True)
    parser.add_argument('--fixed_id', type=int, default=0)
    parser.add_argument('--dataroot', type=str,
                        default='data', help='path to dataset')

    # Optimization arguments
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--seed', type=int, default=2019, help='manual seed')

    # Checkpointing and Logging arguments
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--log_iter_every', type=int, default=100)
    parser.add_argument('--log_epoch_every', type=int, default=1)
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--user', type=str, default='wangkuan')

    # GAN-gd
    parser.add_argument('--init_rs', type=int, default=0)
    parser.add_argument('--gan_gd_lr', type=float, default=0.02)
    parser.add_argument('--gan_gd_m', type=float, default=0.9)
    parser.add_argument('--gan_gd_wd', type=float, default=0)
    parser.add_argument('--gan_gd_lambda', type=float, default=100)
    parser.add_argument('--gan_gd_prior_lambda', type=float, default=1)
    parser.add_argument('--gan_gd_steps', type=int, default=1500)
    parser.add_argument('--gan_gd_init_scale', type=float, default=1)
    parser.add_argument('--prior_model', type=str,
                        default='disc', choices=['disc', 'lep', 'tep', '0'])
    parser.add_argument('--lep_path', type=str, default='')

    # Dev
    parser.add_argument('--db', type=int, default=0)
    parser.add_argument('--overwrite', type=int, default=1)
    args = parser.parse_args()

    if not args.overwrite and os.path.exists(args.output_dir):
        # Check if the previous experiment finished.
        if os.path.exists(os.path.join(args.output_dir, 'after_rs_100_samples.pt')):
            sys.exit(0)

    # Discs
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir, 'sample_pt'))
    mkdir(os.path.join(args.output_dir, 'viz_sample'))
    args.jobid = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else -1
    utils.save_args(args, os.path.join(args.output_dir, 'args.json'))

    # Global Config
    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    print(socket.gethostname())

    main(args)
