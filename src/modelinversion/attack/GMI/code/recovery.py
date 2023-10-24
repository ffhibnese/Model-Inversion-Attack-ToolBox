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
from ....metrics.knn import get_knn_dist
from ....metrics.fid.fid import calc_fid
from ....utils import Tee
from .generator import *
from .generator import Generator
from .utils import log_sum_exp, save_tensor_images


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


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

def inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False,
              num_seeds=5, save_dir='', device='cpu'):
    save_img_dir = os.path.join(save_dir, 'all_imgs')
    success_dir = os.path.join(save_dir, 'success_imgs')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    iden = iden.view(-1).long().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))
    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        z = torch.randn(bs, 100).to(device).float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).to(device).float()

        for i in range(iter_times):
            fake = G(z)
            if improved == True:
                _, label = D(fake)
            else:
                label = D(fake)

            out = T(fake).result

            if z.grad is not None:
                z.grad.data.zero_()

            if improved:
                Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
            # Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
            else:
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
            for i in range(bs):
                gt = iden[i].item()
                sample = G(z)[i]
                all_img_class_path = os.path.join(save_img_dir, str(gt))
                if not os.path.exists(all_img_class_path):
                    os.makedirs(all_img_class_path)
                save_tensor_images(sample.detach(),
                                   os.path.join(all_img_class_path, "attack_iden_{}_{}.png".format(gt, r_idx)))

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = G(z)[i]
                    success_img_class_path = os.path.join(success_dir, str(gt))
                    if not os.path.exists(success_img_class_path):
                        os.makedirs(success_img_class_path)
                    save_tensor_images(best_img.detach(), os.path.join(success_img_class_path,
                                                                       "{}_attack_iden_{}_{}.png".format(itr, gt,
                                                                                                         int(no[i]))))
                    no[i] += 1
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
    

def gmi_attack(target_name, eval_name, ckpt_dir, dataset_name, result_dir, batch_size=60, train_gan_target_name = 'vgg16', device='cpu'):

    
    assert eval_name == 'facenet'
    
    save_dir = os.path.join(result_dir, f'{dataset_name}_{target_name}')
    os.makedirs(save_dir, exist_ok=True)
    Tee(f'{save_dir}/attack.log', 'w')

    print("=> creating model ...")


    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    G = Generator(z_dim)
    # G = torch.nn.DataParallel(G)
        
    # if is_kedmi:
    #     D = MinibatchDiscriminator()
    #     path_G = os.path.join(ckpt_dir, 'KEDMI', f'{dataset_name}_VGG16_KED_MI_G.tar')
    #     path_D = os.path.join(ckpt_dir, 'KEDMI', f'{dataset_name}_VGG16_KED_MI_D.tar')
    # else:
    D = DGWGAN(3)
    path_G = os.path.join(ckpt_dir, 'GMI', f'{dataset_name}_{train_gan_target_name.upper()}_GMI_G.tar')
    path_D = os.path.join(ckpt_dir, 'GMI', f'{dataset_name}_{train_gan_target_name.upper()}_GMI_D.tar')

    # D = torch.nn.DataParallel(D)
    ckp_G = torch.load(path_G, map_location=device)
    G.load_state_dict(ckp_G['state_dict'], strict=True)
    ckp_D = torch.load(path_D, map_location=device)
    D.load_state_dict(ckp_D['state_dict'], strict=True)
    
    # torch.save({'state_dict': G.module.state_dict()}, os.path.join(ckpt_dir, 'GMI', f'{dataset_name}_VGG16_GMI_G.tar'))
    # torch.save({'state_dict': D.module.state_dict()}, os.path.join(ckpt_dir, 'GMI', f'{dataset_name}_VGG16_GMI_D.tar'))
    
    # exit()
    

    if target_name.startswith("vgg16"):
        T = VGG16(1000)
        path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'VGG16_88.26.tar')
    elif target_name.startswith('ir152'):
        T = IR152(1000)
        path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'IR152_91.16.tar')
    elif target_name == "facenet64":
        T = FaceNet64(1000)
        path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'FaceNet64_88.50.tar')
    T = (T).to(device)
    ckp_T = torch.load(path_T, map_location=device)['state_dict']
    T.load_state_dict(ckp_T, strict=True)

    # Load evaluation model
    E = FaceNet(1000)
    E = (E).to(device)
    path_E = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'FaceNet_95.88.tar')
    ckp_E = torch.load(path_E, map_location=device)['state_dict']
    E.load_state_dict(ckp_E, strict=True)

    ############         attack     ###########
    print("=> Begin attacking ...")

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    for i in range(1):
        iden = torch.from_numpy(np.arange(batch_size))

        # evaluate on the first 300 identities only
        for idx in range(5):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            # if is_kedmi:
            #     acc, acc5, var, var5 = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100,
            #                                           iter_times=1500, clip_range=1, improved=True,
            #                                           num_seeds=5, save_dir=save_dir)
            # else:
            acc, acc5, var, var5 = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100,
                                                iter_times=1500, clip_range=1, improved=False,
                                                save_dir=save_dir, device=device)

            iden = iden + batch_size
            aver_acc += acc / 5
            aver_acc5 += acc5 / 5
            aver_var += var / 5
            aver_var5 += var5 / 5

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                            aver_acc5,
                                                                                                            aver_var,
                                                                                                            aver_var5))

    
    print("=> Calculate the KNN Dist.")
    knn_dist = get_knn_dist(E, os.path.join(save_dir, 'all_imgs'), os.path.join(ckpt_dir, 'PLGMI', "celeba_private_feats"), resolution=112)
    print("KNN Dist %.2f" % knn_dist)

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(save_dir, "success_imgs"),
                   private_img_path= os.path.join(ckpt_dir, 'PLGMI', "datasets", "celeba_private_domain"),
                   batch_size=batch_size)
    print("FID %.2f" % fid)
