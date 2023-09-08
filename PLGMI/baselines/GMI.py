import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as tvls
from torch.autograd import grad
from torch.nn import BCELoss

import dataloader
import utils
from classify import *
from discri import DGWGAN
from generator import Generator
from utils import *


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


if __name__ == "__main__":
    global opt

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, default='ffhq')
    parser.add_argument('--save_dir', type=str, default='GMI_Baseline')
    opt = parser.parse_args()
    print(opt)

    save_model_dir = os.path.join(opt.save_dir, opt.dataset_name)
    save_img_dir = os.path.join(save_model_dir, 'images_preview')
    log_path = os.path.join(save_model_dir, 'train_logs')
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    log_file = "train_logs.txt"
    utils.Tee(os.path.join(log_path, log_file), 'w')

    file = "./config/" + opt.dataset_name + ".json"
    args = load_json(json_file=file)

    file_path = args['dataset']['gan_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    name_list, label_list, image_list = utils.load_image_list(args, file_path, mode='gan')
    print("load image list ", len(image_list))
    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan", name_list=name_list,
                                                label_list=label_list, image_list=image_list)

    G = Generator(z_dim)
    DG = DGWGAN(3)

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))

        torch.save({'state_dict': G.state_dict()}, os.path.join(save_model_dir, opt.dataset_name + "_GMI_G.tar"))
        torch.save({'state_dict': DG.state_dict()}, os.path.join(save_model_dir, opt.dataset_name + "_GMI_D.tar"))

        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)),
                               nrow=8)
