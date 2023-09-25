import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from discri import DGWGAN
from generator import Generator

# 冻结与解冻模型参数
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

# WGAN-GP特有的梯度惩罚机制
def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

# 训练结果保存路径
save_img_dir = "../result/imgs_celeba_gan_0.004"
save_model_dir= "../result/models_celeba_gan_0.004"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "GAN0.004.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')

if __name__ == "__main__":
    
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)

    # 打开数据集处理的json文件，加载数据集和待训练的GAN
    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = init_dataloader(args, file_path, batch_size, mode="gan")

    # 载入GAN
    G = Generator(z_dim)
    DG = DGWGAN(3)
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    # 训练
    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            step += 1
            imgs = imgs[0].cuda()
            bs = imgs.size(0)
            
            # 先训练鉴别器
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

            # 再train G

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
        print("Epoch:%d \t Time:%.2f" % (epoch, interval))

        # 记录GAN的优化训练过程
        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)), nrow = 8)
        
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "celeba_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "celeba_D.tar"))

