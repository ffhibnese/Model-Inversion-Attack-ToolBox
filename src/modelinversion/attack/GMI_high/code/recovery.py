import logging
import numpy as np
import os
import random
import statistics
import time
from tqdm import tqdm

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
from ....foldermanager import FolderManager
from ..config import GMIAttackConfig

def reparameterize(mu, std):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """

    # std = torch.exp(0.5 * std)
    eps = torch.randn_like(std)

    return eps * std + mu

def get_reg_loss(featureT,fea_mean, fea_std):
    
    fea_reg = reparameterize(fea_mean, fea_std)
    fea_reg = fea_mean.repeat(featureT.shape[0],1)
    loss_reg = torch.mean((featureT - fea_reg).pow(2))
    # print('loss_reg',loss_reg)
    return loss_reg

def get_iden_loss(imgs, labels, T: BaseTargetModel, augment_models: list[BaseTargetModel]=None, use_logit_loss=False, T_feature_means=None, T_feature_stds=None, reg_coef=0.1):
    if augment_models is None:
        augment_models = []
    iden_loss = 0
    reg_loss = 0
    if use_logit_loss:
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    # loss += loss_fn()
    T_res = T(imgs)
    iden_loss += loss_fn(T_res.result, labels)
    if T_feature_means is not None and T_feature_stds is not None:
        reg_loss += reg_coef * get_reg_loss(T_res.feat[-1], T_feature_means, T_feature_stds)
    
    for aug_model in augment_models:
        res = aug_model(imgs).result
        iden_loss += loss_fn(res, labels)
    
    return iden_loss / (len(augment_models) + 1) + reg_loss
    
        


def inversion(config: GMIAttackConfig, G, D, T, E, iden, folder_manager: FolderManager, augment_models: list[BaseTargetModel] = None, use_logit_loss=False, T_feature_means=None, T_feature_stds=None, reg_coef=0.1):
    
    device = config.device
    momentum = config.momentum
    clip_range = config.clip_range
    lr = config.lr

    iden = iden.view(-1).long().to(device)
    # criterion = nn.NLLLoss() if use_nll_loss else nn.CrossEntropyLoss()
    # criterion = criterion.to(device)
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()
    
    

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, config.gen_num_per_target))
    for random_seed in range(config.gen_num_per_target):
        tf = time.time()
        r_idx = random_seed
        set_random_seed(random_seed)

        z = torch.randn(bs, 100).to(device).float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).to(device).float()

        for i in tqdm(range(config.iter_times)):
            fake = G(z)
            label = D(fake)

            # out = T(fake).result

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()

            # Iden_Loss = criterion(out, iden)
            Iden_Loss = get_iden_loss(fake, iden, T, augment_models, use_logit_loss, T_feature_means, T_feature_stds, reg_coef)

            Total_Loss = Prior_Loss + config.lamda * Iden_Loss

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
                    eval_prob = E(fake_img).result
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

        with torch.no_grad():
            fake = G(z)
            eval_prob = E(fake_img).result
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            
            samples = G(z)
            for i in range(bs):
                gt = iden[i].item()
                sample = samples[i]
                
                folder_manager.save_result_image(sample, gt, save_tensor=True)

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    # flag[i] = 1
                    best_img = samples[i]
                    folder_manager.save_result_image(best_img, gt, folder_name='success_imgs')
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

    # return acc, acc_5, acc_var, acc_var5
    return {
        'acc': acc,
        'acc5': acc_5,
        'acc_var': acc_var,
        'acc5_var': acc_var5
    }