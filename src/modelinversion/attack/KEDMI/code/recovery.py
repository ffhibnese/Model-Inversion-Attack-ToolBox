import logging
import numpy as np
import os
import random
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
from torch.autograd import Variable

from . import utils
from ....models import *
from .discri import *
from ....metrics.fid.fid import calc_fid
from ....utils import Tee
from .generator import *
from .generator import Generator
from .utils import log_sum_exp, save_tensor_images


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def dist_inversion(
    G,
    D,
    T,
    E,
    iden,
    folder_manager,
    lr=2e-2,
    momentum=0.9,
    lamda=100,
    iter_times=1500,
    clip_range=1,
    num_seeds=5,
    device='cpu',
):
    # save_img_dir = os.path.join(save_dir, 'all_imgs')
    # success_dir = os.path.join(save_dir, 'success_imgs')
    # os.makedirs(save_img_dir, exist_ok=True)
    # os.makedirs(success_dir, exist_ok=True)

    iden = iden.view(-1).long().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    no = torch.zeros(bs)  # index for saving all success attack images

    tf = time.time()

    # NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.ones(bs, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(solver, 1800, gamma=0.1)

    for i in range(iter_times):
        z = reparameterize(mu, log_var).to(device)
        fake = G(z)

        _, label = D(fake)

        out = T(fake).result

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(
            log_sum_exp(label)
        )
        Iden_Loss = criterion(out, iden)

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()

        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i + 1) % 300 == 0:
            with torch.no_grad():
                fake_img = G(z.detach())
                eval_prob = E(fake_img).result
                eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                print(
                    "Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(
                        i + 1, Prior_Loss_val, Iden_Loss_val, acc
                    )
                )

    interval = time.time() - tf
    print("Time:{:.2f}".format(interval))

    with torch.no_grad():
        res = []
        res5 = []
        seed_acc = torch.zeros((bs, 5))
        for random_seed in range(num_seeds):
            tf = time.time()
            z = reparameterize(mu, log_var).to(device)
            fake = G(z)
            # score = T(fake).result
            eval_prob = E(fake).result
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                sample = fake[i]

                folder_manager.save_result_image(sample, gt)

                if eval_iden[i].item() == gt:
                    seed_acc[i, random_seed] = 1
                    cnt += 1
                    best_img = G(z)[i]
                    folder_manager.save_result_image(
                        best_img, gt, folder_name='success_imgs'
                    )

                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print(
                "Time:{:.2f}\tSeed:{}\tAcc:{:.2f}\t".format(
                    interval, random_seed, cnt * 1.0 / bs
                )
            )
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)

            torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print(
        "Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(
            acc, acc_5, acc_var, acc_var5
        )
    )

    return acc, acc_5, acc_var, acc_var5
