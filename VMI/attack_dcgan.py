from __future__ import print_function
import sys
import argparse
import os
import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils

import utils
from utils import mkdir, gaussian_logp
from csv_logger import CSVLogger, plot_csv
from main_aux import save_checkpoint, maybe_load_checkpoint
from experimental import AttackExperiment
from likelihood_model import ReparameterizedMVN, FlowMiner, ReparameterizedGMM2


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, lsm, target):
        true_dist = torch.zeros_like(lsm)
        true_dist.fill_(self.smoothing / (self.n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * lsm, dim=self.dim))


class Miner(nn.Module):
    def __init__(self, nz, nz0, nh):
        super(Miner, self).__init__()
        self.nz = nz
        self.nz0 = nz0
        self.nh = nh

        layers_ = [
            nn.Linear(nz0, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nh, nz)
        ]
        self.main = nn.Sequential(*layers_)

    def forward(self, x):
        return self.main(x).squeeze(-1).squeeze(-1)


class MineGAN(nn.Module):
    def __init__(self, miner, generator):
        super(MineGAN, self).__init__()
        self.nz = miner.nz0
        self.is_conditional = generator.is_conditional
        self.miner = miner
        self.generator = generator

    def forward(self, z0, c=None):
        z = self.miner(z0.squeeze(-1).squeeze(-1))
        if c is not None:  # AuxGAN
            x = self.generator(z, c)
        else:  # GAN
            x = self.generator(z)
        return x


def main(args):
    args.ckpt = os.path.join(args.output_dir, "ckpt.pt")

    # db config
    if args.db:
        pass

    # backward compat

    # Experiment setup
    experiment = AttackExperiment(args.exp_config, device, args.db,
                                  fixed_id=args.fixed_id, run_target_feat_eval=args.run_target_feat_eval)
    target_logsoftmax = experiment.target_logsoftmax
    target_dataset = experiment.target_dataset
    target_eval_runner = experiment.target_eval_runner
    generator = experiment.generator
    nclass = experiment.target_dataset['nclass']
    gan_method = experiment.gan_method

    if args.method == 'minegan':
        miner = ReparameterizedMVN(generator.nz).to(device)
        generator = MineGAN(miner, generator)
    elif args.method == 'gmm':
        miner = ReparameterizedGMM2(generator.nz, args.gmm_n_components).to(device)
        generator = MineGAN(miner, generator)
    elif args.method == 'flow':
        miner = FlowMiner(generator.nz, args.flow_permutation,
                          args.flow_K, args.flow_glow, args.flow_coupling, args.flow_L, args.flow_use_actnorm).to(device)
        generator = MineGAN(miner, generator)

    # Opt
    optimizerG = optim.SGD(miner.parameters(), lr=args.lr,
                           momentum=0.9, weight_decay=args.wd)

    # Logging
    iteration_fieldnames = ['global_iteration', 'loss', 'train_target_acc']
    iteration_logger = CSVLogger(every=args.log_iter_every,
                                 fieldnames=iteration_fieldnames,
                                 filename=os.path.join(
                                     args.output_dir, 'iteration_log.csv'),
                                 resume=args.resume)
    epoch_fieldnames = ['global_iteration',
                        'eval-acc-marginal',
                        'eval-frechet-marginal',
                        'eval-feature-l2-dist-marginal',
                        'eval-feature-cos-sim-marginal',
                        'eval-top5_acc-marginal',
                        ]
    if args.run_target_feat_eval:
        epoch_fieldnames += [
            'eval-precision@5-marginal',
            'eval-recall@5-marginal',
            'eval-precision@10-marginal',
            'eval-recall@10-marginal',
        ]
    epoch_logger = CSVLogger(every=args.log_epoch_every,
                             fieldnames=epoch_fieldnames,
                             filename=os.path.join(
                                 args.output_dir, 'epoch_log.csv'),
                             resume=args.resume)

    # Check for ckpt
    ckpt = maybe_load_checkpoint(args)
    if ckpt is not None:
        start_epoch = ckpt['epoch']
        optimizerG.load_state_dict(ckpt['optimizerG'])
        generator.load_state_dict(ckpt['generator'])
    else:
        start_epoch = 0

    patience_count = 0
    best_marginal_acc = 0
    fixed_noise = torch.randn(500, generator.nz, 1, 1, device=device)

    attack_criterion = LabelSmoothingLoss(
        nclass, smoothing=args.attack_labelsmooth)

    save_model_epochs = [int(e) for e in args.save_model_epochs.split(
        ',')] if len(args.save_model_epochs) > 0 else []
    for epoch in range(start_epoch, args.epochs):
        noises = torch.randn(1000, generator.nz, 1, 1, device='cpu')
        # Ckpt
        state = {
            "optimizerG": optimizerG.state_dict(),
            "generator": generator.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(args, state)
        # Save Models
        if epoch in save_model_epochs:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f'generator_{epoch}.pt'))
            torch.save(miner.state_dict(), os.path.join(args.output_dir, f'miner_{epoch}.pt'))

        if epoch > 0 and epoch % args.save_samples_every == 0:
            with torch.no_grad():
                fake = generator(fixed_noise)
            torch.save(fake[:args.n_save_samples], os.path.join(args.output_dir, f'samples_e{epoch}.pt'))

        # Evaluate
        # - Sample conditionally
        all_ys = torch.arange(1000)
        fakes = []
        for start in range(0, 1000, 100):
            with torch.no_grad():
                noise = noises[start:start + 100].to(device)
                if gan_method == 'dcgan_aux':
                    fake_y = torch.ones((100,)) * args.fixed_id
                    fake_y_onehot = torch.eye(nclass)[fake_y.long()].to(device)
                    fake = generator(noise, fake_y_onehot)
                else:
                    if generator.is_conditional:
                        fake_y = all_ys[start:start + 100].to(device)
                        fake = generator(noise, fake_y)
                    else:
                        fake = generator(noise)
                fakes.append(fake)
        fakes = torch.cat(fakes)

        # - Run eval
        epoch_log_dict = {'global_iteration': epoch}
        name = 'marginal'
        fake = fakes[:100]
        fake_y = args.fixed_id * torch.ones(len(fake)).to(device).long()
        D = target_eval_runner.evaluate(fake, fake_y, None)

        for field in D:
            if not ("eval-" + field + f"-{name}" in epoch_fieldnames):
                continue
            epoch_log_dict["eval-" + field + f"-{name}"] = D[field]
        epoch_logger.writerow(epoch_log_dict)
        if len(epoch_log_dict) > 1:
            plot_csv(epoch_logger.filename, os.path.join(
                args.output_dir, 'epoch_plots.jpeg'))

        # Maybe exit
        if epoch_log_dict['eval-acc-marginal'] > best_marginal_acc:
            patience_count = 0
            best_marginal_acc = epoch_log_dict['eval-acc-marginal']
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print("Patience exceeded, exiting")
                sys.exit(0)

        # Visualize samples
        if epoch % args.viz_every == 0:
            def _viz_with_corresponding_preds(fake, fpath):
                with torch.no_grad():
                    preds = target_logsoftmax(fake / 2 + .5).max(1)[1]
                real_target = []
                for c in preds.cpu():
                    real_target.append(
                        target_dataset['X_train'][[target_dataset['Y_train'] == c]][0])
                real_target = torch.stack(real_target)

                preds = target_eval_runner.get_eval_preds(fake)
                real_eval = []
                for c in preds.cpu():
                    real_eval.append(target_dataset['X_train'][[
                                     target_dataset['Y_train'] == c]][0])
                real_eval = torch.stack(real_eval)
                realgrid_target = vutils.make_grid(
                    real_target[:100], nrow=10, padding=4, pad_value=1, normalize=True)
                realgrid_eval = vutils.make_grid(
                    real_eval[:100], nrow=10, padding=4, pad_value=1, normalize=True)
                fakegrid = vutils.make_grid(
                    fake.cpu()[:100], nrow=10, padding=4, pad_value=1, normalize=True)
                fig, axs = plt.subplots(1, 3, figsize=(20, 12))
                axs[0].imshow(np.transpose(
                    realgrid_eval.cpu().numpy(), (1, 2, 0)), interpolation='bilinear')
                axs[0].set_title('Real Eval pred')
                axs[1].imshow(np.transpose(fakegrid.cpu().numpy(),
                                           (1, 2, 0)), interpolation='bilinear')
                axs[1].set_title('Samples')
                axs[2].imshow(np.transpose(realgrid_target.cpu(
                ).numpy(), (1, 2, 0)), interpolation='bilinear')
                axs[2].set_title('Real Target pred')
                for ax in axs:
                    plt.subplot(ax)
                    plt.tight_layout()
                    plt.grid()
                    plt.xticks([])
                    plt.yticks([])
                plt.savefig(fpath, bbox_inches='tight',
                            pad_inches=0, format='jpeg')
            # Marginal samples
            with torch.no_grad():
                if gan_method == 'dcgan_aux':
                    fake_y = torch.ones((len(fixed_noise),)) * args.fixed_id
                    fake_y_onehot = torch.eye(nclass)[fake_y.long()].to(device)
                    fake = generator(fixed_noise, fake_y_onehot)
                else:
                    fake = generator(fixed_noise).detach()
            fpath = f'{args.output_dir}/viz_sample/sample_e{epoch:03d}_marginal.jpeg'
            if experiment.config['data']['name'] == 'celeba':
                _viz_with_corresponding_preds(fake[:100], fpath)
            else:
                fake = fake[:100]
                fakegrid = vutils.make_grid(
                    fake.cpu()[:100], nrow=10, padding=4, pad_value=1, normalize=True)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(np.transpose(
                    fakegrid.cpu().numpy(), (1, 2, 0)), interpolation='bilinear')
                plt.tight_layout()
                plt.grid()
                plt.xticks([])
                plt.yticks([])
                plt.savefig(fpath, bbox_inches='tight',
                            pad_inches=0, format='jpeg')
            # Save Sample Tensors
            if epoch % 10 == 0:
                torch.save(fake[:50].cpu(), os.path.join(args.output_dir, 'samples_pt', f'e{epoch:03d}.pt'))

        # Train loop
        generator.train()
        pbar = tqdm(range(0, 10000, args.batchSize), desc='Train loop')
        for i in pbar:
            generator.zero_grad()

            # Sample from G
            noise = torch.randn(
                args.batchSize, generator.nz, 1, 1, device=device)
            if gan_method == 'dcgan_aux':
                fake_y = torch.ones((args.batchSize,)) * args.fixed_id
                fake_y_onehot = torch.eye(nclass)[fake_y.long()].to(device)
                fake = generator(noise, fake_y_onehot)
            else:
                if generator.is_conditional:
                    fake_y = torch.randint(1000, (args.batchSize,)).to(device)
                    fake = generator(noise, fake_y)
                else:
                    fake = generator(noise)

            # Compute loss
            lsm = target_logsoftmax(fake / 2 + .5)
            fake_y = args.fixed_id * \
                torch.ones(args.batchSize).to(device).long()
            loss_attack = 0
            if args.lambda_attack > 0:
                # loss_attack = -lsm.gather(1, fake_y.view(-1,1)).mean()
                loss_attack = attack_criterion(lsm, fake_y)
            train_target_acc = (lsm.max(1)[1] == fake_y).float().mean().item()

            loss_kl = 0
            # if True:
            if args.lambda_kl > 0:
                if args.method == 'minegan':
                    C = miner.L @ miner.L.T
                    logdetcov = torch.logdet(C)
                    samples = miner(torch.randn(
                        1000, miner.nz0).to(device))
                    loss_kl = -.5 * logdetcov + .5 * \
                        (torch.norm(samples, p=2, dim=[-1])).pow(2).mean()
                else:
                    # KL(Flow || N(0,1))
                    # E_{x ~ Flow}[ log Flow(x) - log N(x; 0,1)]
                    samples = miner(torch.randn(
                        args.batchSize, miner.nz0).to(device))
                    loss_kl = torch.mean(miner.logp(
                        samples) - gaussian_logp(torch.zeros_like(samples), torch.zeros_like(samples), samples).sum(-1))

            loss = (args.lambda_attack * loss_attack
                    + args.lambda_kl * loss_kl)

            loss.backward()
            optimizerG.step()

            # Logging
            pbar.set_postfix_str(s=f'Loss: {loss.item():.2f}, Acc: {train_target_acc:.3f}', refresh=True)

            if i % args.log_iter_every == 0:
                stats_dict = {
                    'global_iteration': iteration_logger.time,
                    'loss': loss.item(),
                    'train_target_acc': train_target_acc
                }
                iteration_logger.writerow(stats_dict)
                plot_csv(iteration_logger.filename, os.path.join(
                    args.output_dir, 'iteration_plots.jpeg'))

            iteration_logger.time += 1


if __name__ == '__main__':
    import socket
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--exp_config', type=str, required=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_model_epochs', type=str, default='')
    parser.add_argument('--method', type=str,
                        default='finetune', choices=['minegan', 'flow', 'gmm'])
    parser.add_argument('--run_target_feat_eval', type=int, default=0)
    parser.add_argument('--attack_labelsmooth', type=float, default=0)
    # Miner
    parser.add_argument('--miner_nh', type=int, default=100)
    parser.add_argument('--miner_z0', type=int, default=50)
    parser.add_argument('--miner_init_std', type=float, default=0.2)
    # EWC
    parser.add_argument('--fixed_id', type=int, default=0)
    parser.add_argument('--ewc_type', type=str, default='fisher')
    parser.add_argument('--lambda_attack', type=float, default=1)
    parser.add_argument('--lambda_kl', type=float, default=0)
    parser.add_argument('--lambda_miner_entropy', type=float, default=0)
    parser.add_argument('--prior_model', type=str, default='disc',
                        choices=['disc', 'lep', 'tep', '0', 'hep'])
    parser.add_argument('--lep_path', type=str, default='')
    parser.add_argument('--flow_permutation', type=str,
                        default='shuffle', choices=['shuffle', 'reverse'])
    parser.add_argument('--flow_K', type=int, default=5)
    parser.add_argument('--flow_glow', type=int, default=0)
    parser.add_argument('--flow_coupling', type=str, default='additive', choices= ['additive', 'affine', 'invconv'])
    parser.add_argument('--flow_L', type=int, default=1)
    parser.add_argument('--flow_use_actnorm', type=int, default=1)
    # GMM
    parser.add_argument('--gmm_n_components', type=int, default=1)

    # Optimization arguments
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float,
                        default=0.5, help='beta1 for adam')
    parser.add_argument('--wd', type=float, default=0., help='wd for adam')
    parser.add_argument('--seed', type=int, default=2019, help='manual seed')

    # Checkpointing and Logging arguments
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--save_samples_every', type=int, default=10000)
    parser.add_argument('--log_iter_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=10)
    parser.add_argument('--log_epoch_every', type=int, default=1)
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--user', type=str, default='wangkuan')

    parser.add_argument('--n_save_samples', type=int, default=100)

    # Dev
    parser.add_argument('--db', type=int, default=0)
    args = parser.parse_args()

    if not args.overwrite and os.path.exists(args.output_dir):
        # Check if the previous experiment ran for more than 10 epochs.
        if os.path.exists(os.path.join(args.output_dir, 'epoch_log.csv')):
            df = pandas.read_csv(os.path.join(
                args.output_dir, 'epoch_log.csv'))
            if len(df) > 10:
                sys.exit(0)

    # Discs
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir, 'viz_sample'))
    mkdir(os.path.join(args.output_dir, 'samples_pt'))
    args.jobid = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else -1
    args.host = socket.gethostname()
    utils.save_args(args, os.path.join(args.output_dir, 'args.json'))

    # Global Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    main(args)
