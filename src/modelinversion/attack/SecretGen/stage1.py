from models import *
from discri import *
from losses import completion_network_loss, noise_loss
from tgt_models.resnet152 import ResNet152
from tgt_models.vgg16 import VGG16
from tgt_models.vit import CONFIGS, VisionTransformer
from utils import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
import random
import os
import os.path as osp
import argparse
import numpy as np
import json
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

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

    c_path = "./premodels/Context_G_xxx.tar"
    i_path = "./premodels/identity_G_xxx.tar"
    ckp_c = torch.load(c_path)
    ckp_i = torch.load(i_path)
    load_my_state_dict(CNet, ckp_c['state_dict'])
    load_my_state_dict(IdenG, ckp_i['state_dict'])
    
    own_state = net.state_dict()
    C_list, I_list = [], []
    
    for n, p in CNet.named_parameters():
        C_list.append([n, p])
    pos = 0
    for name, p in net.named_parameters():
        if name.split('.')[1] == "ContextNetwork":
            print(name)
            own_state[name].copy_(C_list[pos][1].data)
            pos += 1
    
    for n, p in IdenG.named_parameters():
        I_list.append([n, p])
    pos = 0
    for name, p in net.named_parameters():
        if name.split('.')[1] == "IdentityGenerator":
            print(name)
            own_state[name].copy_(I_list[pos][1].data)
            pos += 1


def test_model(test_set, net, iter_times):
    global mpv
    with torch.no_grad():
        #print(len(test_set))
        x = sample_random_batch(test_set, batch_size=32).to(device)
        
        img_size, bs = x.size(2), x.size(0)
        mask = get_mask(img_size, bs, opt.mask)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z1 = torch.randn(bs, z_dim).cuda()
        output1 = net((inp, z1))
        output1 = x - x * mask + output1 * mask

        z2 = torch.randn(bs, z_dim).cuda()
        output2 = Net((inp, z2))
        output2 = x - x * mask + output2 * mask

        imgs = torch.cat((x.cpu(), x_mask.cpu(), output1.cpu(), output2.cpu()), dim=0)
        imgpath = os.path.join(result_img_dir, 'm1_step%d.png' % iter_times)
        save_tensor_images(imgs, imgpath, nrow=bs)

def gradient_penalty_dl(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    __, o = DL(z)
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
    
    __, o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='vgg16', choices=['resnet152', 'vgg16', 'ir50', 'ViT-B_16'], help='type of model to use')
parser.add_argument('--bb', action='store_true', help='whether blackbox access')
parser.add_argument('--mask', type=str, default='center', choices=['center', 'face'], help='type of mask')
opt = parser.parse_args()
print(opt)

batch_size = 64
if opt.bb:
    result_img_dir = f"./result/blackbox_{opt.mask}/img"
    result_model_dir = f"./result/blackbox_{opt.mask}/model"
else:
    result_img_dir = f"./result/{opt.name}_{opt.mask}/img"
    result_model_dir = f"./result/{opt.name}_{opt.mask}/model"
pretrain_dir = "./premodels"
os.makedirs(result_img_dir, exist_ok=True)
os.makedirs(result_model_dir, exist_ok=True)


'''
img_path = osp.expanduser('~/CelebA/celeba/img_align_celeba/')
identity_file = osp.expanduser('~/CelebA/celeba/identity_CelebA.txt')

with open(identity_file) as f:
    lines = f.readlines()

id2file = {}
for line in lines:
    file, id = line.strip().split()
    id = int(id)
    if id in id2file.keys():
        id2file[id].append(file)
    else:
        id2file[id] = [file]

thres = 25
id2file_cleaned = {}
for key in id2file.keys():
    if len(id2file[key]) > thres:
        id2file_cleaned[key] = id2file[key]

img_list = []

for key in sorted(id2file_cleaned.keys())[:2000]:
    for file in id2file_cleaned[key][:20]:
        img_list.append(file)

mpv = np.zeros(shape=(3,))     
pbar = tqdm(total=len(img_list), desc='computing mean pixel value for training dataset...')
for img_name in img_list:
    path = img_path + "/" + img_name
    img = Image.open(path)
    x = np.array(img, dtype=np.float32) / 255.
    mpv += x.mean(axis=(0,1))
    pbar.update()
mpv /= len(img_list)
pbar.close()
print(mpv)
input()
'''

mpv = np.array([0.5189, 0.4346, 0.3886])
mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).cuda()
data_set, data_loader = init_dataloader(batch_size, split='pub')
test_set, test_loader = init_dataloader(batch_size, split='pub-dev')



# ================================================
# Training Phase 1
# ================================================
lr = 0.004
z_dim = 100

Net = InversionNet().cuda()
DL = DLWGAN().cuda()
DG = DGWGAN().cuda()

# DL.load_state_dict(torch.load(osp.join(result_model_dir, 'DL_xxx.tar'))['state_dict'])
# DG.load_state_dict(torch.load(osp.join(result_model_dir, 'DG_xxx.tar'))['state_dict'])
# Net.load_state_dict(torch.load(osp.join(result_model_dir, 'Inversion_xxx.tar'))['state_dict'])

Net = torch.nn.DataParallel(Net)
DL = torch.nn.DataParallel(DL)#.cuda()
DG = torch.nn.DataParallel(DG)#.cuda()

dl_optimizer = torch.optim.Adam(DL.parameters(), lr=lr, betas=(0.5, 0.999))
dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
net_optimizer = Adam(Net.parameters(), lr=lr, betas=(0.5,0.999))

#print(Net)

# loadnet(Net)


num_classes = 1000
if not opt.bb:
    if opt.name == 'resnet152':
        V = ResNet152(num_classes=num_classes)
    elif opt.name == 'vgg16':
        V = VGG16(num_classes=num_classes)
    elif opt.name == 'ir50':
        V = FaceNet(num_classes=num_classes)
    elif opt.name == 'ViT-B_16':
        V = VisionTransformer(CONFIGS[opt.name], num_classes=num_classes)

    if opt.bb:
        V.load_state_dict(torch.load(osp.join('premodels', f'{opt.name}-pub.tar'))['state_dict'])
    else:
        V.load_state_dict(torch.load(osp.join('premodels', f'{opt.name}-pri.tar'))['state_dict'])
else:
    V = FaceNet(num_classes=num_classes)
    V.feature.load_state_dict(torch.load('premodels/ir50.pth'))

V = torch.nn.DataParallel(V).cuda()

for param in V.parameters():
    param.requires_grad = False

if opt.name == 'ir50' or opt.bb:
    scale_fnV = low2high112
elif opt.name == 'ViT-B_16':
    scale_fnV = low2high224
else:
    scale_fnV = None


start_epoch = 0
epochs = 100
# training
epoch_bar = tqdm(range(start_epoch, epochs))

total_iter = 0
writer = SummaryWriter(log_dir=osp.join("logs", f'{opt.name}_stage1'))

for epoch in epoch_bar:
    
    dl = AverageMeter()
    dg = AverageMeter()
    gan = AverageMeter()
    re = AverageMeter()
    diff = AverageMeter()
    st = time.time()
    cnt = 0

    pbar = tqdm(data_loader)
    for imgs, _, _ in pbar:

        x = imgs.cuda()
        img_size = x.size(2)
        bs = x.size(0)

        if bs < 8:
            continue

        # train dl
        
        freeze(DG)
        freeze(Net)
        unfreeze(DL)
        
        mask = get_mask(img_size, bs, opt.mask)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z = torch.randn(bs, z_dim).cuda()
        output = Net((inp, z))
        output = x - x * mask + output * mask

        hole_area = gen_hole_area((ld_input_size, ld_input_size), (x.shape[3], x.shape[2]))
        fake_crop = crop(output, hole_area)
        real_crop = crop(x, hole_area)

        __, r_logit = DL(real_crop)
        __, f_logit = DL(fake_crop)
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

        
        mask = get_mask(img_size, bs, opt.mask)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z = torch.randn(bs, z_dim).cuda()
        output = Net((inp, z))
        output = x - x * mask + output * mask

        __, r_logit = DG(x)
        __, f_logit = DG(output)
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

        mask = get_mask(img_size, bs, opt.mask)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z1 = torch.randn(bs, z_dim).cuda()
        output1 = Net((inp, z1))
        output1 = x - x * mask + output1 * mask

        hole_area = gen_hole_area((ld_input_size, ld_input_size), (x.shape[3], x.shape[2]))
        fake_crop = crop(output1, hole_area)
        
        fl1, logit_dl = DL(fake_crop)
        fg1, logit_dg = DG(output1)
        # calculate g_loss
        gan_loss = (- logit_dl.mean() - logit_dg.mean()) / 2
        re_loss = completion_network_loss(x, output1, mask)

        z2 = torch.randn(bs, z_dim).cuda()
        output2 = Net((inp, z2))
        output2 = x - x * mask + output2 * mask

        fake_crop = crop(output2, hole_area)
        fl2, __ = DL(fake_crop)
        fg2, __ = DG(output2)

        if scale_fnV is not None:
            diff_loss = noise_loss(V, scale_fnV(output1), scale_fnV(output2))
        else:
            diff_loss = noise_loss(V, output1, output2)
        diff_loss = diff_loss / torch.mean(torch.abs(z2 - z1))

        loss = gan_loss - diff_loss * 0.5
        gan.update(gan_loss.detach().cpu().numpy(), bs)
        re.update(re_loss.detach().cpu().numpy(), bs)
        if not opt.bb:
            diff.update(diff_loss.detach().cpu().numpy(), bs)
        
        pbar.set_description("gan_loss:{:.3f}, re_loss:{:.3f}, diff_loss:{:.3f}".format(gan_loss, re_loss, diff_loss))

        net_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()

        writer.add_scalars('G', {'loss': loss, 'gan_loss': gan_loss, 'diff_loss': diff_loss}, total_iter)
        writer.add_scalars('D', {'D_global': dg_loss, 'D_local': dl_loss}, total_iter)

        total_iter += 1
        

    interval = time.time() - st
    st = time.time()
    test_model(test_set, Net, epoch)
    epoch_bar.set_description("Epoch:{}\tTime:{:.2f}\tgan:{:.2f}\tre:{:.2f}\tdiff:{:.2f}".format(
        epoch, interval, float(gan.avg), float(re.avg), float(diff.avg)
        ))       
    
    torch.save({'state_dict':Net.module.state_dict()}, result_model_dir + '/' + "Inversion_xxx.tar")
    torch.save({'state_dict':DG.module.state_dict()}, result_model_dir + '/' + "DG_xxx.tar")
    torch.save({'state_dict':DL.module.state_dict()}, result_model_dir + '/' + "DL_xxx.tar")

    
