from collections import OrderedDict
import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation

from . import losses as L
from . import utils
from ...models import *
# from ...metrics import get_knn_dist, calc_fid
# from ...metrics import knn
# from ...metrics.fid.fid import calc_fid

from .models.generators.resnet64 import ResNetGenerator
from .utils import save_tensor_images
from .config import PlgmiAttackConfig
from ...utils import Tee
from ...utils import set_random_seed




def inversion(args, G, T, E, iden, folder_manager, lr=2e-2, iter_times=1500, num_seeds=5):
    # save_img_dir = os.path.join(args.save_dir, 'all_imgs')
    # success_dir = os.path.join(args.save_dir, 'success_imgs')
    # os.makedirs(save_img_dir, exist_ok=True)
    # os.makedirs(success_dir, exist_ok=True)
    
    set_random_seed(42)

    bs = iden.shape[0]
    iden = iden.view(-1).long().to(args.device)

    G.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed

        set_random_seed(random_seed)

        z = utils.sample_z(
            bs, args.gen_dim_z, args.device, args.gen_distribution
        )
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(iter_times):
            fake = G(z, iden)
            out1 = T(aug_list(fake)).result
            out2 = T(aug_list(fake)).result

            if z.grad is not None:
                z.grad.data.zero_()

            if args.inv_loss_type == 'ce':
                inv_loss = L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
            elif args.inv_loss_type == 'margin':
                inv_loss = L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
            elif args.inv_loss_type == 'poincare':
                inv_loss = L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake_img = G(z, iden)
                    eval_prob = E(fake_img).result
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        with torch.no_grad():
            fake = G(z, iden)
            # score = T(fake).result
            eval_prob = E(fake).result
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                # sample = G(z, iden)[i]
                sample = fake[i]
                folder_manager.save_result_image(sample, gt)

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = fake[i]
                    folder_manager.save_result_image(best_img, gt, folder_name='success_imgs')
                    
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            if args.device == 'cuda':
                torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5

from dataclasses import dataclass

@dataclass
class PlgmiArgs:
    taregt_name: str
    eval_name: str
    save_dir: str
    path_G: str
    device: str
    
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_num_features: int = 64
    gen_dim_z: int = 128
    gen_bottom_width: int = 4
    gen_distribution: str = 'normal'
    
