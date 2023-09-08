import os, torch, random, json, time
import numpy as np
from PIL import Image
from torch.autograd import grad
from discri import DLWGAN, DGWGAN
from torch.optim import Adadelta, Adam
from generator import CompletionNetwork
from losses import completion_network_loss
from utils import gen_hole_area, save_tensor_images, init_dataloader, load_params
from utils import get_input_mask, crop, AverageMeter, sample_random_batch, get_center_mask


device = "cuda"
bsize = 64
result_img_dir = "./result/imgs_ganset_context"
result_model_dir = "./result/models_ganset_context"
os.makedirs(result_img_dir, exist_ok=True)
os.makedirs(result_model_dir, exist_ok=True)
ld_input_size, cn_input_size = 32, 64
num_test = 16

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True) 

def test_model(test_set, model_cn, iter_times):
    global mpv
    with torch.no_grad():
        #print(len(test_set))
        x = sample_random_batch(test_set, batch_size=num_test).to(device)
        img_size, bs = x.size(2), x.size(0)
        mask = get_center_mask(img_size, bs)
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model_cn(input)
        imgs = torch.cat((x.cpu(), x_mask.cpu(), output.cpu()), dim=0)
        imgpath = os.path.join(result_img_dir, 'step%d.png' % iter_times)
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

if __name__ == "__main__":
    mpv = np.zeros(shape=(3,))
    dataset_name = "celeba"
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)
    file_path = "train_file_path"
    test_file_path = "test_file_path"
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
    mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).to(device)
    
    data_set, data_loader = init_dataloader(args, file_path, bsize)
    test_set, test_loader = init_dataloader(args, test_file_path, bsize)
    
    # ================================================
    # Training Phase 1
    # ================================================
    lr = 0.0002
    G = CompletionNetwork().cuda()
    g_optimizer = Adam(G.parameters())

    G = torch.nn.DataParallel(G)#.cuda()

    DL = DLWGAN().cuda()
    DG = DGWGAN().cuda()

    DL = torch.nn.DataParallel(DL)#.cuda()
    DG = torch.nn.DataParallel(DG)#.cuda()

    dl_optimizer = torch.optim.Adam(DL.parameters(), lr=lr, betas=(0.5, 0.999))
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))

    # training
    # 20000 100
    # 10000 200
    # 5000 400
    for epoch in range(200):
        
        dl = AverageMeter()
        dg = AverageMeter()
        gan = AverageMeter()
        re = AverageMeter()
        st = time.time()
        cnt = 0

        for imgs in data_loader:
            x = imgs.cuda()
            img_size = x.size(2)
            bs = x.size(0)
            
            # train dl
            
            freeze(DG)
            freeze(G)
            unfreeze(DL)
            
            mask = get_center_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            output = G(inp)

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
            freeze(G)
            unfreeze(DG)

            mask = get_center_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            
            output = G(inp)

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
            unfreeze(G)

            mask = get_center_mask(img_size, bs)
            x_mask = x - x * mask + mpv * mask
            inp = torch.cat((x_mask, mask), dim=1)
            output = G(inp)
            hole_area = gen_hole_area((ld_input_size, ld_input_size), (x.shape[3], x.shape[2]))
            fake_crop = crop(output, hole_area)
            
            __, logit_dl = DL(fake_crop)
            __, logit_dg = DG(output)
            # calculate g_loss
            gan_loss = (- logit_dl.mean() - logit_dg.mean()) / 2
            re_loss = completion_network_loss(x, output, mask)
            loss = gan_loss * 20 + re_loss 
            gan.update(gan_loss.detach().cpu().numpy(), bs)
            re.update(re_loss.detach().cpu().numpy(), bs)
            # print("gan_loss:{:.3f}\tre_loss:{:.3f}".format(gan_loss, re_loss))

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()
            
            
        interval = time.time() - st
        st = time.time()
        if epoch % 20 == 0:
            test_model(test_set, G, epoch)
        print("Epoch:{}\tTime:{:.2f}\tdl:{:.2f}\tdg:{:.2f}\tgan:{:.2f}\tre:{:.2f}".format(
                epoch, interval, dl.avg, dg.avg, gan.avg, re.avg))       
        
        torch.save({'state_dict':G.state_dict()}, result_model_dir + '/' + "Context_G.tar")
        torch.save({'state_dict':DG.state_dict()}, result_model_dir + '/' + "DG.tar")
        torch.save({'state_dict':DL.state_dict()}, result_model_dir + '/' + "DL.tar")

                
    
