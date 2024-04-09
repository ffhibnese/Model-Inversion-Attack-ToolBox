# StyleGAN
import re
import sys

sys.path.append('../stylegan2-ada-pytorch')  # noqa: E702

from typing import List
import legacy
import dnnlib
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
from utils import gaussian_logp
from csv_logger import CSVLogger, plot_csv
from main import save_checkpoint, maybe_load_checkpoint
from experimental import AttackExperiment
from likelihood_model import (
    ReparameterizedMVN,
    MixtureOfRMVN,
    MixtureOfIndependentRMVN,
    FlowMiner,
    LayeredFlowMiner,
    MixtureOfGMM,
)


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


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


class MineGAN(nn.Module):
    def __init__(self, miner, Gmapping):
        super(MineGAN, self).__init__()
        self.nz = miner.nz0
        self.miner = miner
        self.Gmapping = Gmapping

    def forward(self, z0):
        z = self.miner(z0)
        w = self.Gmapping(z, None)
        return w


class LayeredMineGAN(nn.Module):
    def __init__(self, miner, Gmapping):
        super(LayeredMineGAN, self).__init__()
        self.nz = miner.nz0
        self.miner = miner
        self.Gmapping = Gmapping

    def forward(self, z0):
        N, zdim = z0.shape
        z = self.miner(z0)  # (N, zdim) -> (N, l, zdim)
        w = self.Gmapping(z.reshape(-1, zdim), None)  # (N * l, l, zdim)
        w = w[:, 0].reshape(N, -1, zdim)  # (N, l, zdim)
        return w


class IndependentLayeredMineGAN(LayeredMineGAN):
    def __init__(self, miner, Gmapping):
        super(IndependentLayeredMineGAN, self).__init__(miner, Gmapping)

    def forward(self, z0s):
        nl, N, zdim = z0s.shape
        z = self.miner(z0s)  # (l, N, zdim) -> (N, l, zdim)
        w = self.Gmapping(z.reshape(-1, zdim), None)  # (N * l, l, zdim)
        w = w[:, 0].reshape(N, -1, zdim)  # (N, l, zdim)
        return w


def main(args):
    # args.ckpt = f"/checkpoint/{args.user}/{os.environ['SLURM_JOB_ID']}/ckpt.pt"
    args.ckpt = os.path.join(args.output_dir, "ckpt.pt")

    # db config
    if args.db:
        pass

    # backward compat

    # Experiment setup
    experiment = AttackExperiment(
        args.exp_config,
        device,
        args.db,
        fixed_id=args.fixed_id,
        run_target_feat_eval=args.run_target_feat_eval,
    )
    target_logsoftmax = experiment.target_logsoftmax
    target_dataset = experiment.target_dataset
    target_eval_runner = experiment.target_eval_runner
    nclass = experiment.target_dataset['nclass']

    # StyleGAN
    print('Loading networks from "%s"...' % args.network)
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    noise_mode = 'const'

    if args.method == 'minegan':
        miner = ReparameterizedMVN(G.mapping.z_dim).to(device).double()
        minegan_Gmapping = MineGAN(miner, G.mapping)
    elif args.method == 'layeredminegan':
        miner = MixtureOfRMVN(G.mapping.z_dim, G.mapping.num_ws).to(device).double()
        minegan_Gmapping = LayeredMineGAN(miner, G.mapping)
    elif args.method == 'layeredgmm':
        miner = (
            MixtureOfGMM(G.mapping.z_dim, args.gmm_n_components, G.mapping.num_ws)
            .to(device)
            .double()
        )
        minegan_Gmapping = LayeredMineGAN(miner, G.mapping)

    elif args.method == 'flow':
        miner = (
            FlowMiner(
                G.mapping.z_dim,
                args.flow_permutation,
                args.flow_K,
                args.flow_glow,
                args.flow_coupling,
                args.flow_L,
                args.flow_use_actnorm,
            )
            .to(device)
            .double()
        )
        minegan_Gmapping = MineGAN(miner, G.mapping)
    elif args.method == 'layeredflow':
        miner = (
            LayeredFlowMiner(
                G.mapping.z_dim,
                G.mapping.num_ws,
                args.flow_permutation,
                args.flow_K,
                args.flow_glow,
                args.flow_coupling,
                args.flow_L,
                args.flow_use_actnorm,
            )
            .to(device)
            .double()
        )
        minegan_Gmapping = LayeredMineGAN(miner, G.mapping)

    args.l_identity = num_range(args.l_identity)
    identity_mask = torch.zeros(1, G.mapping.num_ws, 1).to(device)
    identity_mask[:, args.l_identity, :] = 1

    # Opt
    optimizerG = optim.SGD(
        miner.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    # Logging
    iteration_fieldnames = ['global_iteration', 'loss', 'train_target_acc']
    iteration_logger = CSVLogger(
        every=args.log_iter_every,
        fieldnames=iteration_fieldnames,
        filename=os.path.join(args.output_dir, 'iteration_log.csv'),
        resume=args.resume,
    )
    epoch_fieldnames = [
        'global_iteration',
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
    epoch_logger = CSVLogger(
        every=args.log_epoch_every,
        fieldnames=epoch_fieldnames,
        filename=os.path.join(args.output_dir, 'epoch_log.csv'),
        resume=args.resume,
    )

    # Check for ckpt
    ckpt = maybe_load_checkpoint(args)
    if ckpt is not None:
        start_epoch = ckpt['epoch']
        optimizerG.load_state_dict(ckpt['optimizerG'])
        miner.load_state_dict(ckpt['miner'])
    else:
        start_epoch = 0

    patience_count = 0
    best_marginal_acc = 0

    fixed_z_nuisance = torch.randn(100, G.z_dim).to(device).double()
    fixed_z_identity = torch.randn(100, G.z_dim).to(device).double()

    attack_criterion = LabelSmoothingLoss(nclass, smoothing=args.attack_labelsmooth)

    save_model_epochs = (
        [int(e) for e in args.save_model_epochs.split(',')]
        if len(args.save_model_epochs) > 0
        else []
    )

    def sample(z_nuisance, z_identity):
        w_nuisance = G.mapping(z_nuisance, None)
        w_identity = minegan_Gmapping(z_identity)
        w = (1 - identity_mask) * w_nuisance + identity_mask * w_identity
        x = G.synthesis(w, noise_mode=noise_mode)
        return x

    if args.eval:
        sd = torch.load(args.ckpt_path)
        with torch.no_grad():
            z_nu = torch.randn(100, G.z_dim).to(device).double()
            z_id = torch.randn(100, G.z_dim).to(device).double()
            fake = sample(z_nu, z_id)
            vutils.save_image(fake * 0.5 + 0.5, 'dbb.jpeg')
            miner.load_state_dict(sd)
            fake = sample(z_nu, z_id)
            vutils.save_image(fake * 0.5 + 0.5, 'dba.jpeg')

        sys.exit(0)

    for epoch in range(start_epoch, args.epochs):
        # Ckpt
        state = {
            "optimizerG": optimizerG.state_dict(),
            "miner": miner.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(args, state)
        # Save Models
        if epoch in save_model_epochs:
            torch.save(
                miner.state_dict(), os.path.join(args.output_dir, f'miner_{epoch}.pt')
            )

        if epoch > 0 and epoch % args.save_samples_every == 0:
            with torch.no_grad():
                fake = sample(fixed_z_nuisance, fixed_z_identity)
            torch.save(
                fake[: args.n_save_samples],
                os.path.join(args.output_dir, f'samples_e{epoch}.pt'),
            )

        # Evaluate
        if epoch % args.eval_every == 0:
            fakes = []
            for start in range(0, 1000, 100):
                with torch.no_grad():
                    z_nu = torch.randn(100, G.z_dim).to(device).double()
                    z_id = torch.randn(100, G.z_dim).to(device).double()
                    fake = sample(z_nu, z_id)
                    fakes.append(fake)
            fakes = torch.cat(fakes)

            # - Run eval
            epoch_log_dict = {'global_iteration': epoch}
            fake = fakes[:100]
            name = 'marginal'
            fake_y = args.fixed_id * torch.ones(len(fake)).to(device).long()

            with torch.no_grad():
                D = target_eval_runner.evaluate(fake, fake_y, None)

            for field in D:
                if not ("eval-" + field + f"-{name}" in epoch_fieldnames):
                    continue
                epoch_log_dict["eval-" + field + f"-{name}"] = D[field]
            epoch_logger.writerow(epoch_log_dict)
            if len(epoch_log_dict) > 1:
                plot_csv(
                    epoch_logger.filename,
                    os.path.join(args.output_dir, 'epoch_plots.jpeg'),
                )

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
                    preds = target_logsoftmax(fake / 2 + 0.5).max(1)[1]
                real_target = []
                for c in preds.cpu():
                    real_target.append(
                        target_dataset['X_train'][[target_dataset['Y_train'] == c]][0]
                    )
                real_target = torch.stack(real_target)

                preds = target_eval_runner.get_eval_preds(fake)
                real_eval = []
                for c in preds.cpu():
                    real_eval.append(
                        target_dataset['X_train'][[target_dataset['Y_train'] == c]][0]
                    )
                real_eval = torch.stack(real_eval)
                realgrid_target = vutils.make_grid(
                    real_target[:100], nrow=10, padding=4, pad_value=1, normalize=True
                )
                realgrid_eval = vutils.make_grid(
                    real_eval[:100], nrow=10, padding=4, pad_value=1, normalize=True
                )
                fakegrid = vutils.make_grid(
                    fake.cpu()[:100], nrow=10, padding=4, pad_value=1, normalize=True
                )
                fig, axs = plt.subplots(1, 3, figsize=(20, 12))
                axs[0].imshow(
                    np.transpose(realgrid_eval.cpu().numpy(), (1, 2, 0)),
                    interpolation='bilinear',
                )
                axs[0].set_title('Real Eval pred')
                axs[1].imshow(
                    np.transpose(fakegrid.cpu().numpy(), (1, 2, 0)),
                    interpolation='bilinear',
                )
                axs[1].set_title('Samples')
                axs[2].imshow(
                    np.transpose(realgrid_target.cpu().numpy(), (1, 2, 0)),
                    interpolation='bilinear',
                )
                axs[2].set_title('Real Target pred')
                for ax in axs:
                    plt.subplot(ax)
                    plt.tight_layout()
                    plt.grid()
                    plt.xticks([])
                    plt.yticks([])
                plt.savefig(fpath, bbox_inches='tight', pad_inches=0, format='jpeg')

            # Marginal samples
            with torch.no_grad():
                fake = sample(fixed_z_nuisance, fixed_z_identity).clamp(-1, 1)
            _viz_with_corresponding_preds(
                fake, f'{args.output_dir}/viz_sample/sample_e{epoch:03d}_marginal.jpeg'
            )
            if epoch % 10 == 0:
                torch.save(
                    fake[:50].cpu(),
                    os.path.join(args.output_dir, 'samples_pt', f'e{epoch:03d}.pt'),
                )

        # Train loop
        miner.train()
        pbar = tqdm(range(0, 10000, args.batchSize), desc='Train loop')
        for i in pbar:
            optimizerG.zero_grad()

            # Sample from G
            z_nu = torch.randn(args.batchSize, G.z_dim).to(device).double()
            z_id = torch.randn(args.batchSize, G.z_dim).to(device).double()
            fake = sample(z_nu, z_id).clamp(-1, 1)

            # Compute loss
            lsm = target_logsoftmax(fake / 2 + 0.5)
            fake_y = args.fixed_id * torch.ones(args.batchSize).to(device).long()
            loss_attack = 0
            if args.lambda_attack > 0:
                loss_attack = attack_criterion(lsm, fake_y)
            train_target_acc = (lsm.max(1)[1] == fake_y).float().mean().item()

            loss_miner_entropy = 0
            if args.lambda_miner_entropy > 0:
                loss_miner_entropy = -miner.entropy()

            loss_kl = 0
            # if True:
            if args.lambda_kl > 0 and i % args.kl_every == 0:
                if args.method == 'minegan':
                    mu = miner.m
                    C = miner.L @ miner.L.T
                    logdetcov = torch.logdet(C)
                    samples = miner(torch.randn(1000, miner.nz0).to(device).double())
                    loss_kl = (
                        -0.5 * logdetcov
                        + 0.5 * (torch.norm(samples, p=2, dim=[-1])).pow(2).mean()
                    )
                elif args.method == 'layeredminegan':
                    for mvn in miner.mvns:
                        mu = mvn.m
                        C = mvn.L @ mvn.L.T
                        logdetcov = torch.logdet(C)
                        samples = mvn(torch.randn(1000, mvn.nz0).to(device).double())
                        loss_kl += (
                            -0.5 * logdetcov
                            + 0.5 * (torch.norm(samples, p=2, dim=[-1])).pow(2).mean()
                        )
                elif args.method == 'flow':
                    # KL(Flow || N(0,1))
                    # E_{x ~ Flow}[ log Flow(x) - log N(x; 0,1)]
                    samples = miner(torch.randn(1000, miner.nz0).to(device).double())
                    loss_kl = torch.mean(
                        miner.logp(samples)
                        - gaussian_logp(
                            torch.zeros_like(samples),
                            torch.zeros_like(samples),
                            samples,
                        ).sum(-1)
                    )
                elif args.method == 'layeredgmm':
                    for gmm in miner.gmms:
                        samples = gmm(
                            torch.randn(args.batchSize, gmm.nz0).to(device).double()
                        )
                        loss_kl += torch.mean(
                            gmm.logp(samples)
                            - gaussian_logp(
                                torch.zeros_like(samples),
                                torch.zeros_like(samples),
                                samples,
                            ).sum(-1)
                        )
                    loss_kl /= len(miner.gmms)
                elif args.method == 'layeredflow':
                    # 1/L * \sum_l KL(Flow_l || N(0,1))
                    for flow in miner.flow_miners:
                        samples = flow(
                            torch.randn(args.batchSize, flow.nz0).to(device).double()
                        )
                        loss_kl += torch.mean(
                            flow.logp(samples)
                            - gaussian_logp(
                                torch.zeros_like(samples),
                                torch.zeros_like(samples),
                                samples,
                            ).sum(-1)
                        )
                    loss_kl /= len(miner.flow_miners)

            loss = (
                args.lambda_attack * loss_attack
                + args.lambda_miner_entropy * loss_miner_entropy
                + args.lambda_kl * loss_kl
            )

            loss.backward()
            optimizerG.step()

            # Logging
            pbar.set_postfix_str(
                s=f'Loss: {loss.item():.2f}, Acc: {train_target_acc:.3f}', refresh=True
            )

            if i % args.log_iter_every == 0:
                stats_dict = {
                    'global_iteration': iteration_logger.time,
                    'loss': loss.item(),
                    'train_target_acc': train_target_acc,
                }
                iteration_logger.writerow(stats_dict)
                plot_csv(
                    iteration_logger.filename,
                    os.path.join(args.output_dir, 'iteration_plots.jpeg'),
                )

            iteration_logger.time += 1


if __name__ == '__main__':
    import socket

    parser = argparse.ArgumentParser()
    # Eval
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='')
    # StyleGAN
    parser.add_argument('--l_identity', type=str, default='0-6')
    parser.add_argument('--network', type=str, required=True)
    #
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--exp_config', type=str, required=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_model_epochs', type=str, default='')
    parser.add_argument(
        '--method',
        type=str,
        default='minegan',
        choices=['minegan', 'layeredminegan', 'flow', 'layeredflow', 'layeredgmm'],
    )
    parser.add_argument('--run_target_feat_eval', type=int, default=0)
    parser.add_argument('--attack_labelsmooth', type=float, default=0)
    # Miner
    parser.add_argument('--miner_nh', type=int, default=100)
    parser.add_argument('--miner_z0', type=int, default=50)
    parser.add_argument('--miner_init_std', type=float, default=0.2)
    parser.add_argument(
        '--flow_permutation',
        type=str,
        default='shuffle',
        choices=['shuffle', 'reverse'],
    )
    parser.add_argument('--flow_K', type=int, default=5)
    parser.add_argument('--flow_glow', type=int, default=0)
    parser.add_argument(
        '--flow_coupling',
        type=str,
        default='additive',
        choices=['additive', 'affine', 'invconv'],
    )
    parser.add_argument('--flow_L', type=int, default=1)
    parser.add_argument('--flow_use_actnorm', type=int, default=1)
    parser.add_argument('--gmm_n_components', type=int, default=1)
    # EWC
    parser.add_argument('--fixed_id', type=int, default=0)
    parser.add_argument('--ewc_type', type=str, default='fisher')
    parser.add_argument('--lambda_weight_reg', type=float, default=1)
    parser.add_argument('--lambda_attack', type=float, default=1)
    parser.add_argument('--lambda_prior', type=float, default=0)
    parser.add_argument('--lambda_miner_entropy', type=float, default=0)
    parser.add_argument('--lambda_kl', type=float, default=0)
    parser.add_argument(
        '--prior_model',
        type=str,
        default='disc',
        choices=['disc', 'lep', 'tep', '0', 'hep'],
    )

    # Optimization arguments
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=1000, help='number of epochs to train for'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0002, help='learning rate, default=0.0002'
    )
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--wd', type=float, default=0.0, help='wd for adam')
    parser.add_argument('--seed', type=int, default=2019, help='manual seed')
    parser.add_argument('--kl_every', type=int, default=1)

    # Checkpointing and Logging arguments
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--save_samples_every', type=int, default=10000)
    parser.add_argument('--log_iter_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--log_epoch_every', type=int, default=1)
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--user', type=str, default='wangkuan')

    parser.add_argument('--n_save_samples', type=int, default=100)

    # Dev
    parser.add_argument('--db', type=int, default=0)
    parser.add_argument('--dbg', type=int, default=0)
    args = parser.parse_args()

    if not args.overwrite and os.path.exists(args.output_dir):
        # # Check if the previous experiment ran for more than 10 epochs.
        # if os.path.exists(os.path.join(args.output_dir, 'epoch_log.csv')):
        #     df = pandas.read_csv(os.path.join(
        #         args.output_dir, 'epoch_log.csv'))
        #     if len(df) > 10:
        #         sys.exit(0)

        # Check if ckpt 50 exists
        if os.path.exists(os.path.join(args.output_dir, 'miner_20.pt')):
            sys.exit(0)

    # Discs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'viz_sample'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_pt'), exist_ok=True)
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
