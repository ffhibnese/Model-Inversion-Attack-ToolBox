from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion, dist_inversion
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


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



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--device', type=str, default='4,5,6,7', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--improved_flag', action='store_true', default=False, help='use improved k+1 GAN')
    parser.add_argument('--dist_flag', action='store_true', default=False, help='use distributional recovery')
    args = parser.parse_args()
    logger = get_logger()

    logger.info(args)
    logger.info("=> creating model ...")

    print("=> Using improved GAN:", args.improved_flag)
   
    
    
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    G = Generator(z_dim)
    G = torch.nn.DataParallel(G).cuda()
    if args.improved_flag == True:
        D = MinibatchDiscriminator()
        path_G = './improvedGAN/improved_celeba_G.tar'
        path_D = './improvedGAN/improved_celeba_D.tar'
    else:
        D = DGWGAN(3)
        path_G = './improvedGAN/celeba_G.tar'
        path_D = './improvedGAN/celeba_D.tar'
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)

    if args.model.startswith("VGG16"):
        T = VGG16(1000)
        path_T = './target_model/target_ckp/VGG16_88.26.tar'
    elif args.model.startswith('IR152'):
        T = IR152(1000)
        path_T = './target_model/target_ckp/IR152_91.16.tar'
    elif args.model == "FaceNet64":
        T = FaceNet64(1000)
        path_T = './target_model/target_ckp/FaceNet64_88.50.tar'

    
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = './target_model/target_ckp/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)


    ############         attack     ###########
    logger.info("=> Begin attacking ...")

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    for i in range(1):
        iden = torch.from_numpy(np.arange(60))

        # evaluate on the first 300 identities only
        for idx in range(5):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            if args.dist_flag == True:
                acc, acc5, var, var5 = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=2400, clip_range=1, improved=args.improved_flag, num_seeds=5)
            else:
                acc, acc5, var, var5 = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=2400, clip_range=1, improved=args.improved_flag)
            
            iden = iden + 60
            aver_acc += acc / 5
            aver_acc5 += acc5 / 5
            aver_var += var / 5
            aver_var5 += var5 / 5

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc, aver_acc5, aver_var, aver_var5))

    