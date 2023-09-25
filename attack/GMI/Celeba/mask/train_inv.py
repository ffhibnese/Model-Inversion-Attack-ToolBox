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
import os
import numpy as np
import json

ld_input_size, cn_input_size = 32, 64

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True) 

def loadnet(net):
    CNet = ContextNetwork().cuda()
    IdenG = IdentityGenerator().cuda()
    CNet = torch.nn.DataParallel(CNet)
    IdenG = torch.nn.DataParallel(IdenG)

    c_path = "./result_model/Context_G.tar"
    i_path = "./result_model/identity_G.tar"
    ckp_c = torch.load(c_path)['state_dict']
    ckp_i = torch.load(i_path)['state_dict']
    
    load_my_state_dict(CNet, ckp_c)
    load_my_state_dict(IdenG, ckp_i)
    
    own_state = net.state_dict()
    C_list, I_list = [], []
    
    for n, p in CNet.named_parameters():
        C_list.append([n, p])
    pos = 0
    for name, p in net.named_parameters():
        if name.split('.')[1] == "ContextNetwork":
            own_state[name].copy_(C_list[pos][1].data)
            pos += 1
    
    for n, p in IdenG.named_parameters():
        I_list.append([n, p])
    pos = 0
    for name, p in net.named_parameters():
        if name.split('.')[1] == "IdentityGenerator":
            own_state[name].copy_(I_list[pos][1].data)
            pos += 1


def test_model(test_set, net, iter_times):
    global mpv
    with torch.no_grad():
        #print(len(test_set))
        x = sample_random_batch(test_set, batch_size=32).to(device)
        
        img_size, bs = x.size(2), x.size(0)
        mask = get_mask(img_size, bs)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z = torch.randn(bs, z_dim).cuda()
        output = net((inp, z))
        #completed = poisson_blend(x, output, mask)
        imgs = torch.cat((x.cpu(), x_mask.cpu(), output.cpu()), dim=0)
        imgpath = os.path.join(result_img_dir, 'imgs_step%d.png' % iter_times)
        save_tensor_images(imgs, imgpath, nrow=bs)

def gradient_penalty_dl(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DL(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def gradient_penalty_dg(x, y):
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

lr = 0.004
z_dim = 100

if __name__ == "__main__":
    dataset_name = "celeba"
    batch_size = 64
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)
    result_img_dir = "./result/mask_img_reg"
    result_model_dir = "./result/mask_model_reg"
    os.makedirs(result_img_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    file_path = "train_file_path"
    test_file_path = "test_file_path"
    mpv = np.zeros(shape=(3,))
    '''
    pbar = tqdm(total=len(img_list), desc='computing mean pixel value for training dataset...')
    for img_name in img_list:
        if img_name.endswith(".png"):
            path = img_path + "/" + img_name
            img = Image.open(path)
            x = np.array(img, dtype=np.float32) / 255.
            mpv += x.mean(axis=(0,1))
            pbar.update()
    mpv /= len(img_list)
    pbar.close()
    '''
    mpv = np.array([0.5061, 0.4254, 0.3828])
    mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).cuda()
    data_set, data_loader = init_dataloader(args, file_path, batch_size)
    test_set, test_loader = init_dataloader(args, test_file_path, batch_size)

    Net = InversionNet().cuda()
    Net = torch.nn.DataParallel(Net)

    DL = DLWGAN().cuda()
    DG = DGWGAN().cuda()

    DL = torch.nn.DataParallel(DL)#.cuda()
    DG = torch.nn.DataParallel(DG)#.cuda()

    dl_optimizer = torch.optim.Adam(DL.parameters(), lr=lr, betas=(0.5, 0.999))
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    net_optimizer = Adam(Net.parameters(), lr=lr, betas=(0.5,0.999))

    model_name = "VGG16"

    if model_name.startswith("VGG16"):
        V = VGG16(1000)
    elif model_name.startswith('IR152'):
        V = IR152(1000)
    elif model_name == "FaceNet64":
        V = FaceNet64(1000)
    
    V = torch.nn.DataParallel(V).cuda()
    path_V = "attack_model/" + model_name + ".tar"
    ckp_V = torch.load(path_V)
    load_my_state_dict(V, ckp_V['state_dict'])

    epochs = 100
    # training
    for epoch in range(epochs):
        
        dl = AverageMeter()
        dg = AverageMeter()
        gan = AverageMeter()
        re = AverageMeter()
        diff = AverageMeter()
        st = time.time()
        cnt = 0

        for imgs in data_loader:

            x = imgs.cuda()
            img_size = x.size(2)
            bs = x.size(0)

            if bs < 8:
                continue

            # train dl
            
            freeze(DG)
            freeze(Net)
            unfreeze(DL)
            
            mask = get_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            z = torch.randn(bs, z_dim).cuda()
            output = Net((inp, z))
            output = x - x * mask + output * mask

            hole_area = gen_hole_area((ld_input_size, ld_input_size), (x.shape[3], x.shape[2]))
            fake_crop = crop(output, hole_area)
            real_crop = crop(x, hole_area)

            r_logit = DL(real_crop)
            f_logit = DL(fake_crop)
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            
            gp = gradient_penalty_dl(fake_crop.data, real_crop.data)
            dl_loss = - wd + gp * 10.0
            dl.update(dl_loss.detach().cpu().numpy(), bs)
            
            dl_optimizer.zero_grad()
            dl_loss.backward()
            dl_optimizer.step()
            
            #train dg
            
            freeze(DL)
            freeze(Net)
            unfreeze(DG)

            mask = get_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            z = torch.randn(bs, z_dim).cuda()
            output = Net((inp, z))
            output = x - x * mask + output * mask

            r_logit = DG(x)
            f_logit = DG(output)
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            
            gp = gradient_penalty_dg(x.data, output.data)
            dg_loss = - wd + gp * 10.0
            dg.update(dg_loss.detach().cpu().numpy(), bs)

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()
            
            # train G
            
            freeze(DL)
            freeze(DG)
            unfreeze(Net)

            mask = get_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            z1 = torch.randn(bs, z_dim).cuda()
            output1 = Net((inp, z1))
            output1 = x - x * mask + output1 * mask

            hole_area = gen_hole_area((ld_input_size, ld_input_size), (x.shape[3], x.shape[2]))
            fake_crop = crop(output1, hole_area)
            
            logit_dl = DL(fake_crop)
            logit_dg = DG(output1)
            # calculate g_loss
            gan_loss = (- logit_dl.mean() - logit_dg.mean()) / 2
            re_loss = completion_network_loss(x, output1, mask)

            z2 = torch.randn(bs, z_dim).cuda()
            output2 = Net((inp, z2))
            output2 = x - x * mask + output2 * mask

            diff_loss = noise_loss(V, output1, output2)
            diff_loss = diff_loss / torch.mean(torch.abs(z2 - z1))

            loss = gan_loss - diff_loss * 0.5
            gan.update(gan_loss.detach().cpu().numpy(), bs)
            re.update(re_loss.detach().cpu().numpy(), bs)
            diff.update(diff_loss.detach().cpu().numpy(), bs)
           
            net_optimizer.zero_grad()
            loss.backward()
            net_optimizer.step()

        interval = time.time() - st
        st = time.time()
        if (epoch+1) % 1 ==0:
            test_model(test_set, Net, epoch)
        
        print("Epoch:{}\tTime:{:.2f}\tgan:{:.2f}\tre:{:.2f}\tdiff:{:.2f}".format(
            epoch, interval, float(gan.avg), float(re.avg), float(diff.avg)
            ))       
        
        torch.save({'state_dict':Net.state_dict()}, result_model_dir + '/' + "Inversion.tar")
        torch.save({'state_dict':DG.state_dict()}, result_model_dir + '/' + "DG.tar")
        torch.save({'state_dict':DL.state_dict()}, result_model_dir + '/' + "DL.tar")

        
