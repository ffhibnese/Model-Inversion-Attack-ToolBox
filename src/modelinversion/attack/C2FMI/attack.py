import os
import torch
import random
from torch import optim
from ...utils import FolderManager
from .config import C2FMIConfig
from .code.gan_model import Generator
from .code.models.facenet import Facenet
from .code.models.predict2feature import predict2feature
from .code.reconstruct import inversion_attack

def attack(config: C2FMIConfig):
    save_dir = os.path.join(config.result_dir, f'{config.gan_dataset_name}_{config.target_name}')
    folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, save_dir)
    
    print("=> creating model ...")
    
    # get parameters
    device = config.device
    img_size = config.img_size
    tar_backbone = config.tar_backbone
    tar_classes = config.tar_classes
    emb_backbone = config.emb_backbone
    emb_classes = config.emb_classes
    trunc = config.mask
    emb_backbone = config.emb_backbone
    tar_backbone = config.tar_backbone
    n_mean_latent = config.n_mean_latent
    batch = config.batch_size
    init_lr = config.init_lr
    target_labels = config.target_labels
    step = config.step
    face_shape = config.face_shape
    only_best = config.only_best
    input_latent  = True
    
    # load models
    G = Generator(img_size, 512, 8, channel_multiplier=1)
    T = Facenet(backbone=tar_backbone, num_classes=tar_classes)
    Embed = Facenet(backbone=emb_backbone, num_classes=emb_classes)
    P2f = predict2feature(tar_classes, trunc)
    
    folder_manager.load_state_dict(G, ['C2FMI', config.gan_path], device=device)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=device)
    folder_manager.load_state_dict(Embed, ['C2FMI', config.emb_path], device=device)
    folder_manager.load_state_dict(P2f, ['C2FMI', config.p2f_pth], device=device)
    
    G.to(device)
    T.to(device)
    Embed.to(device)
    P2f.to(device)
    
    G.eval()
    T.eval()
    Embed.eval()
    P2f.eval()
    
    # 获得待优化的初始隐向量
    with torch.no_grad():
        noise_samples = torch.randn(n_mean_latent, 512, device=device)
        latents       = G.style(noise_samples)
        latent_mean   = latents.mean(0)
        latent_std    = (((latents-latent_mean).pow(2).sum() / n_mean_latent) ** 0.5).item()
        latent_in     = torch.zeros((batch, 512)).to(device)
        latent_in.requires_grad = True
    
    noises = None
    optimizer    = optim.Adam([latent_in], lr=init_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    print(f'attack from {target_labels[0]} to {target_labels[-1]}')
    for target_label in target_labels:
        # 为当前batch的每一个隐向量进行初始化
        with torch.no_grad():
            for i in range(batch):
                j   = random.randint(0, n_mean_latent//10-100)
                tmp = latents[2*j:2*(j+1),:].detach().mean(0).clone()
                latent_in[i,:] = tmp

        optimizer    = optim.Adam([latent_in], lr=init_lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

        imgs = inversion_attack(G, T, Embed, P2f, target_label, latent_in, step, optimizer,
                         lr_scheduler, face_shape, img_size, input_latent, tar_classes, trunc, device, only_best)
        folder_manager.temp_cnt = 0
        for k,img in enumerate(imgs):
            folder_manager.save_result_image(img, target_label)