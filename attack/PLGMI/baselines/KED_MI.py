import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as tvls
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.autograd import grad
from torch.nn import BCELoss

from . import dataloader
from . import utils
from models import *
from .discri import DGWGAN, Discriminator, MinibatchDiscriminator
from .generator import Generator
from .utils import *

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


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


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name_T', type=str, default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--dataset_name', type=str, default='celeba')
    parser.add_argument('--save_dir', type=str, default='KED-MI_Baseline')
    opt = parser.parse_args()
    print(opt)

    save_model_dir = os.path.join(opt.save_dir, opt.dataset_name, opt.model_name_T)
    save_img_dir = os.path.join(save_model_dir, 'images_preview')
    log_path = os.path.join(save_model_dir, 'train_logs')
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    log_file = "train_logs.txt"
    utils.Tee(os.path.join(log_path, log_file), 'w')

    file = "./config/" + opt.dataset_name + ".json"
    print("config: ", file)
    args = load_json(json_file=file)
    writer = SummaryWriter(log_path)

    file_path = args['dataset']['gan_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    model_name_T = opt.model_name_T

    if model_name_T.startswith("VGG16"):
        T = VGG16(1000)
        path_T = '../checkpoints/target_model/VGG16_88.26.tar'

    elif model_name_T.startswith('IR152'):
        T = IR152(1000)
        path_T = '../checkpoints/target_model/IR152_91.16.tar'
    elif model_name_T == "FaceNet64":
        T = FaceNet64(1000)
        path_T = '../checkpoints/target_model/FaceNet64_88.50.tar'
    print("Target Model: ", path_T)

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    print("---------------------Training [%s]------------------------------" % model_name)

    utils.print_params(args["dataset"], args[model_name])
    name_list, label_list, image_list = utils.load_image_list(args, file_path, mode='gan')
    print("load image list ", len(image_list))
    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan", name_list=name_list,
                                                label_list=label_list, image_list=image_list)

    G = Generator(z_dim)
    DG = MinibatchDiscriminator()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()

    step = 0

    for epoch in range(epochs):

        start = time.time()
        _, unlabel_loader1 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True,
                                             name_list=name_list, label_list=label_list, image_list=image_list)
        _, unlabel_loader2 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True,
                                             name_list=name_list, label_list=label_list, image_list=image_list)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake = DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel))) - torch.mean(
                log_sum_exp(output_unlabel)) + torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict': G.state_dict()},
                   os.path.join(save_model_dir, opt.dataset_name + "_" + opt.model_name_T + "_KED_MI_G.tar"))
        torch.save({'state_dict': DG.state_dict()},
                   os.path.join(save_model_dir, opt.dataset_name + "_" + opt.model_name_T + "_KED_MI_D.tar"))

        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir,
                                                                 opt.dataset_name + "_" + opt.model_name_T + "_img_{}.png".format(
                                                                     epoch)), nrow=8)
