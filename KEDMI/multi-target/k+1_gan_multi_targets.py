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
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from generator import Generator
from classify import *
from tensorboardX import SummaryWriter
from datetime import datetime
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
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


save_img_dir = "./improvedGAN/imgs_improved_celeba_gan"
save_model_dir= "./improvedGAN/"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./improvedGAN/attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "improvedGAN_celeba.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'
    global args, writer
    
    file = "./config/" + dataset_name + ".json"
    args = load_json(json_file=file)
    writer = SummaryWriter(log_path)

    file_path = args['dataset']['gan_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    ########### Add the multiple target models to train on them ###########
    target_models = []
    target_paths = []

    target_models.append(VGG16(1000))
    target_paths.append('models/target_ckp/VGG16_88.26.tar')

    target_models.append(IR152(1000))
    target_paths.append('models/target_ckp/IR152_91.16.tar')

    target_models.append(FaceNet64(1000))
    target_paths.append('models/target_ckp/FaceNet64_88.50.tar')

    T = []
    print('TRAINING ON TARGETS:\t', target_paths)
    for i in range(len(target_models)):
        model_weight = 1 / len(target_models)
        model = target_models[i]
        model = torch.nn.DataParallel(model).cuda()
        ckp_T = torch.load(target_paths[i])
        model.load_state_dict(ckp_T['state_dict'], strict=False)
        T.append([model, model_weight])

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan")

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
        _, unlabel_loader1 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)

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

            y_prob=[]
            y=[]
            for target_model, model_weight in T:
                y_prob.append([target_model(imgs)[-1], model_weight])
                y.append(torch.argmax(y_prob[-1], dim=1).view(-1))
            

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake =  DG(f_imgs)

            

            loss_lab = []
            for current_y_prob, model_weight in y_prob:
                loss_lab.append(model_weight * softXEnt(output_label, current_y_prob))

            loss_lab = sum(loss_lab)


            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab
            
            acc=[]
            for current_y in y:
                acc.append(torch.mean((output_label.max(1)[1] == current_y).float()))
            
            
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

                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss  

                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f" % (epoch, interval, g_loss),'targets accuracies:', acc)

        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_celeba_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_celeba_D.tar"))

        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow = 8)
