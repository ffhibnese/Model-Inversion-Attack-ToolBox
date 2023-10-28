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
from ....utils import Tee
from .generator import *
from .generator import Generator
from .utils import save_tensor_images
from ....utils import set_random_seed
from ....utils import FolderManager


def inversion(G, D, T, E, iden, folder_manager: FolderManager, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, num_seeds=5, device='cpu'):
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

    # flag = torch.zeros(bs)
    # no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))
    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed
        set_random_seed(random_seed)

        z = torch.randn(bs, 100).to(device).float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).to(device).float()

        for i in range(iter_times):
            fake = G(z)
            label = D(fake)

            out = T(fake).result

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()

            Iden_Loss = criterion(out, iden)

            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if (i + 1) % 300 == 0:
                with torch.no_grad():
                    fake_img = G(z.detach())
                    eval_prob = E(utils.low2high(fake_img, device)).result
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

        with torch.no_grad():
            fake = G(z)
            score = T(fake).result
            eval_prob = E(utils.low2high(fake, device)).result
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            
            samples = G(z)
            for i in range(bs):
                gt = iden[i].item()
                sample = samples[i]
                
                folder_manager.save_result_image(sample, gt)
                # all_img_class_path = os.path.join(save_img_dir, str(gt))
                # if not os.path.exists(all_img_class_path):
                #     os.makedirs(all_img_class_path)
                # save_tensor_images(sample.detach(),
                #                    os.path.join(all_img_class_path, "attack_iden_{}_{}.png".format(gt, r_idx)))

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    # flag[i] = 1
                    best_img = samples[i]
                    folder_manager.save_result_image(best_img, gt)
                    # success_img_class_path = os.path.join(success_dir, str(gt))
                    # if not os.path.exists(success_img_class_path):
                    #     os.makedirs(success_img_class_path)
                    # save_tensor_images(best_img.detach(), os.path.join(success_img_class_path,
                    #                                                    "{}_attack_iden_{}_{}.png".format(itr, gt,
                    #                                                                                      int(no[i]))))
                    # no[i] += 1
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5