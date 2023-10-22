import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

import matplotlib.pylab as plt
import seaborn as sns
import os
import pickle
import math

import utils
import numpy as np

from torch.distributions.normal import Normal
from csv_logger import plot_csv
from tqdm import tqdm
from utils import prepare_random_indices
from DiffAugment_pytorch import DiffAugment
real_label = 1
fake_label = 0
criterion = nn.BCELoss()


def mm(x_train, netG, netD, optimizerG, optimizerD, args, epoch, iteration_logger):
    print('current memory allocated: {}'.format(
        torch.cuda.memory_allocated() / 1024 ** 2))
    device = args.device
    x_index = torch.arange(len(x_train))

    # Shuffle Data
    ridx = np.random.permutation(len(x_index))
    x_train = x_train.clone()[ridx]
    x_index = x_index.clone()[ridx]

    print('current memory allocated: {}'.format(
        torch.cuda.memory_allocated() / 1024 ** 2))

    # Optim
    for i in range(0, len(x_train), args.batchSize):
        netD.zero_grad()
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)
        real_y = x_index[i:i+stop].to(device)

        batch_size = real_x.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        real_x = DiffAugment(real_x / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            real_x += args.d_noise * torch.randn_like(real_x)
        output = netD(real_x, real_y)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        real_acc = (output >= .5).float().mean().item()

        # train with fake
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        fake_y = torch.randint(args.n_conditions, (batch_size,)).to(device)
        fake = netG(noise, fake_y)
        label.fill_(fake_label)
        fake = DiffAugment(fake / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            fake = fake + args.d_noise * torch.randn_like(fake)
        output = netD(fake.detach(), fake_y)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        fake_acc = (output < .5).float().mean().item()
        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())
        # (2) Update G network: maximize log(D(G(z)))

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake, fake_y)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # log performance
        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                       % (epoch, args.epochs, i, len(x_train), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            # Collect fields
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            plot_csv(iteration_logger.filename, os.path.join(
                args.output_dir, 'iteration_plots.jpeg'))

        iteration_logger.time += 1


def l2_aux(x_train, netG, optimizerG, args, epoch, iteration_logger, real_c=None):

    device = args.device
    x_train = x_train.to(device)

    # Shuffle Data
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    c_train = real_c.clone()[ridx]

    # Optim
    for i in range(0, len(x_train), args.batchSize):
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)

        # Extract c from aux dataset
        c = c_train[i:i+stop].to(device)

        # Train with fake
        batch_size = real_x.size(0)
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        fake = netG(noise, c)
        errG = torch.pow(fake.view(batch_size, -1) -
                         real_x.view(batch_size, -1), 2).sum(-1)
        errG = errG.mean()

        # Backprop and step
        netG.zero_grad()
        errG.backward()
        optimizerG.step()

        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] ..  Loss_G(l2): %.4f'
                       % (epoch, args.epochs, i, len(x_train), errG.data))

            # Collect fields
            # - Dummies
            d_loss, real_acc, fake_acc, acc = 0, 0, 0, 0
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            try:
                plot_csv(iteration_logger.filename, os.path.join(
                    args.output_dir, 'iteration_plots.jpeg'))
            except:
                pass

        iteration_logger.time += 1


def dcgan_aux(x_train, netG, netD, optimizerG, optimizerD, args, epoch, iteration_logger, y_train=None, smoothness_extract_feat=lambda x: x, real_c=None):
    assert netD.is_conditional
    device = args.device

    # Shuffle Data
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    c_train = real_c.clone()[ridx]

    # Optim
    for i in range(0, len(x_train), args.batchSize):

        netD.zero_grad()
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)
        c = c_train[i:i+stop].to(device)

        batch_size = real_x.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        real_x = DiffAugment(real_x / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            real_x += args.d_noise * torch.randn_like(real_x)
        output = netD(real_x, c)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        real_acc = (output >= .5).float().mean().item()

        # train with fake
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        fake = netG(noise, c)
        label.fill_(fake_label)

        fake = DiffAugment(fake / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            fake = fake + args.d_noise * torch.randn_like(fake)
        output = netD(fake.detach(), c)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        fake_acc = (output < .5).float().mean().item()
        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())

        # Update G network: maximize log(D(G(z)))

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake, c)
        errG = criterion(output, label)
        if args.l2_aux_reg > 0:
            fake = netG(noise, c)
            real_x = x_train[i:i+stop].to(device)
            # errG = errG + args.l2_aux_reg * torch.pow( fake.view(batch_size, -1) - real_x.view(batch_size, -1), 2).sum(-1).mean()
            errG = args.l2_aux_reg * \
                torch.pow(fake.view(batch_size, -1) -
                          real_x.view(batch_size, -1), 2).sum(-1).mean()

        # Backprop and step
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # log performance
        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                       % (epoch, args.epochs, i, len(x_train), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            # Collect fields
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            try:
                plot_csv(iteration_logger.filename, os.path.join(
                    args.output_dir, 'iteration_plots.jpeg'))
            except:
                pass

        iteration_logger.time += 1


def wgan_aux(x_train, netG, netD, optimizerG, optimizerD, args, epoch, iteration_logger, y_train=None, smoothness_extract_feat=lambda x: x, target_extract_feat=None):
    assert netD.is_conditional
    device = args.device
    x_train = x_train.to(device)

    # Optim
    for i in range(0, len(x_train), args.batchSize):

        netD.zero_grad()
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)

        # Extract c from aux dataset
        c = target_extract_feat(real_x / 2 + .5).detach()

        batch_size = real_x.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        real_x = DiffAugment(real_x / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            real_x += args.d_noise * torch.randn_like(real_x)
        output = netD(real_x, c)
        errD_real = -output.mean()
        errD_real.backward()
        D_x = output.mean().item()
        real_acc = (output >= .5).float().mean().item()

        # train with fake
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        fake = netG(noise, c)
        label.fill_(fake_label)

        fake = DiffAugment(fake / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            fake = fake + args.d_noise * torch.randn_like(fake)
        output = netD(fake.detach(), c)
        errD_fake = output.mean()
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        fake_acc = (output < .5).float().mean().item()
        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())

        # Update G network: maximize log(D(G(z)))

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake, c)
        errG = output.mean()

        # Backprop and step
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # log performance
        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f .. errD_real: %.4f .. errD_fake: %.4f .. errG %.4f'
                       % (epoch, args.epochs, i, len(x_train), errD.data, errG.data, D_x, D_G_z1, D_G_z2, errD_real, errD_fake, errG))

            # Collect fields
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            try:
                plot_csv(iteration_logger.filename, os.path.join(
                    args.output_dir, 'iteration_plots.jpeg'))
            except:
                pass

        iteration_logger.time += 1


def logmeanexp(x, axis):
    return torch.logsumexp(x, axis) - np.log(x.size(axis))


def kplus1gan(x_train, netG, netD, optimizerG, optimizerD, args, epoch, iteration_logger, y_train=None, smoothness_extract_feat=lambda x: x, target_logsoftmax=None):
    """
    logits =
    l_gen = nn.log_sum_exp(output_before_softmax_gen)
    loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
    """
    # assert netG.is_conditional == (y_train is not None)

    # Shuffle Data
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    if y_train is not None:
        y_train = y_train.clone()[ridx]

    device = args.device
    # x_train = x_train.to(device)
    # if y_train  is not None:
    #     y_train = y_train.to(device)

    # Optim
    for i in range(0, len(x_train), args.batchSize):
        netD.zero_grad()
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)
        real_y = y_train[i:i+stop].to(device) if y_train is not None else None

        batch_size = real_x.size(0)

        # Train with real
        real_x = DiffAugment(real_x / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            real_x += args.d_noise * torch.randn_like(real_x)
        # lsm = F.log_softmax(netD.logits(real_x))
        logits = netD.logits(real_x)
        if netG.is_conditional:
            errD_real = F.nll_loss(logits, real_y, reduction='none').mean()
            errD_real += criterion(torch.sigmoid(logmeanexp(logits, 1)),
                                   torch.full((batch_size,), real_label, device=device))
        else:
            errD_real = criterion(torch.sigmoid(logmeanexp(logits, 1)), torch.full(
                (batch_size,), real_label, device=device))

        # Distillation reg
        if args.kplus1_distill_lambda > 0:
            pyx = target_logsoftmax(
                x_train[i:i+stop].to(device) / 2 + .5).exp()
            L_distill = - \
                torch.mean(torch.sum(pyx * F.log_softmax(logits, 1), 1))
            errD_real_opt = errD_real + args.kplus1_distill_lambda * L_distill
            loss_distill = L_distill.item()
        else:
            errD_real_opt = errD_real
            loss_distill = 0
        errD_real_opt.backward()

        # real_acc = (lsm.exp()[:,-1] <= .5).float().mean().item()
        real_acc = (logmeanexp(logits, 1) > 0).float().mean().item()

        # Train with fake
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        if netG.is_conditional:
            fake = netG(noise, real_y)
        else:
            fake = netG(noise)

        fake = DiffAugment(fake / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            fake = fake + args.d_noise * torch.randn_like(fake)
        logits = netD.logits(fake.detach())
        errD_fake = criterion(torch.sigmoid(logmeanexp(logits, 1)), torch.full(
            (batch_size,), fake_label, device=device))
        # lsm = F.log_softmax(netD.logits(fake.detach()))
        # label = netD.n_conditions * torch.ones(len(lsm)).to(device).long()
        # errD_fake = F.nll_loss(lsm, label)
        errD_fake.backward()
        errD = errD_real + errD_fake

        optimizerD.step()
        # fake_acc = (lsm.exp()[:,-1] > .5).float().mean().item()
        fake_acc = (logmeanexp(logits, 1) < 0).float().mean().item()

        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        # lsm = F.log_softmax(netD.logits(fake))
        logits = netD.logits(fake)
        if netG.is_conditional:
            # errG = F.nll_loss(lsm, real_y)
            errG = F.nll_loss(logits, real_y, reduction='none').mean()
            errG += criterion(torch.sigmoid(logmeanexp(logits, 1)),
                              torch.full((batch_size,), real_label, device=device))
        else:
            errG = criterion(torch.sigmoid(logmeanexp(logits, 1)), torch.full(
                (batch_size,), real_label, device=device))

        # Backprop and step
        errG.backward()
        optimizerG.step()

        # log performance
        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f '
                       % (epoch, args.epochs, i, len(x_train), errD.data, errG.data))
            # Maybe compute classification accuracy
            if netG.is_conditional:
                logits = netD.logits(real_x)
                lsm_k = F.log_softmax(logits[:, :netD.n_conditions], 1)
                class_acc = (lsm_k.max(1)[1] == real_y).float().mean().item()

            # Collect fields
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            plot_csv(iteration_logger.filename, os.path.join(
                args.output_dir, 'iteration_plots.jpeg'))

        iteration_logger.time += 1


def dcgan(x_train, netG, netD, optimizerG, optimizerD, args, epoch, iteration_logger, y_train=None, target_extract_feat=lambda x: x):
    device = args.device
    # x_train = x_train.to(device)
    # if netG.is_conditional:
    # y_train = y_train.to(device)

    # Shuffle Data
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    if netG.is_conditional:
        y_train = y_train.clone()[ridx]

    # Optim
    for i in range(0, len(x_train), args.batchSize):
        netD.zero_grad()
        stop = min(args.batchSize, len(x_train[i:]))
        real_x = x_train[i:i+stop].to(device)
        real_y = y_train[i:i+stop].to(device) if netG.is_conditional else None

        batch_size = real_x.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        real_x = DiffAugment(real_x / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            real_x += args.d_noise * torch.randn_like(real_x)
        output = netD(real_x, real_y)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        real_acc = (output >= .5).float().mean().item()

        # train with fake
        if not netG.is_conditional:
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            fake_y = None
        else:
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            # fake_y = torch.randint(args.n_conditions, (batch_size,)).to(device)
            fake_y = real_y
            fake = netG(noise, fake_y)
        label.fill_(fake_label)

        fake = DiffAugment(fake / 2 + .5, args.augment) * 2 - 1
        if args.d_noise > 0:
            fake = fake + args.d_noise * torch.randn_like(fake)
        output = netD(fake.detach(), fake_y)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        fake_acc = (output < .5).float().mean().item()
        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())

        # (2) Update G network: maximize log(D(G(z)))

        netG.zero_grad()
        label.fill_(real_label)
        if args.use_topk:
            # Topk training: https://arxiv.org/pdf/2002.06224.pdf
            output = netD(fake, fake_y)
            # import ipdb;  ipdb.set_trace()
            k = min(args.topk_k, len(output))  # for the last batch < batchsize
            topk_output = torch.topk(output, k=k)[0]
            errG = criterion(topk_output, label[:k])
        else:
            output = netD(fake, fake_y)
            errG = criterion(output, label)

        # Diversity regularization
        if args.lambda_diversity > 0:
            z0 = noise
            z1 = noise + torch.randn_like(noise)
            fake0 = netG(z0)
            fake1 = netG(z1)
            xdiff = torch.pow(
                target_extract_feat(fake0 / 2 + .5).view(batch_size, -1) -
                target_extract_feat(fake1 / 2 + .5).view(batch_size, -1), 2).sum(-1)
            zdiff = torch.pow(z1 - z0, 2).sum(-1)
            # Added a sigmoid for normalizing the magnitude
            diversity_loss = torch.mean(
                torch.sigmoid(- xdiff / (zdiff + 1e-5)))
            errG = errG + args.lambda_diversity * diversity_loss

        # Backprop and step
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # log performance
        if i % args.log_iter_every == 0:
            args.print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                       % (epoch, args.epochs, i, len(x_train), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            # Collect fields
            stats_dict = {'global_iteration': iteration_logger.time}
            for k in iteration_logger.fieldnames:
                if k != 'global_iteration':
                    stats_dict[k] = eval(k)

            iteration_logger.writerow(stats_dict)
            plot_csv(iteration_logger.filename, os.path.join(
                args.output_dir, 'iteration_plots.jpeg'))

        iteration_logger.time += 1
