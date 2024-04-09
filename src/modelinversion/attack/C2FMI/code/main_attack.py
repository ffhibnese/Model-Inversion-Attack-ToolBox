import torch
from torch import nn, optim
from models.facenet import Facenet
from gan_model import Generator, Discriminator
from tqdm import tqdm
from torchvision.transforms import Resize
from torchvision import utils as uts
from torch.nn import functional as F
import random
from models.predict2feature import predict2feature
from utils.DE import Optimizer as post_opt
from utils.DE import DE_c2b_5_bin, DE_c2b_5_bin2


def resize_img_gen_shape(img_gen, trans):
    t_img = trans(img_gen)
    face_input = t_img.clamp(min=-1, max=1).add(1).div(2)
    return face_input


def post_de(latent_in, generator, target_model, target_label, idx):
    x = latent_in[idx]
    optim_DE = post_opt(
        target_model, generator, target_label, direct='gen_figures/DE_og_casia3_incep/'
    )
    # task = DE_c2b_5_bin(optim_DE, max_gen=300, x=x)
    task = DE_c2b_5_bin2(optim_DE, max_gen=300, x=x)
    task.run(disturb=0.00)
    task.get_img(32, only_best=True)


def disturb_latent(latent_in, disturb_strenth):
    latent_n = latent_in
    disturb = torch.randn_like(latent_n) * disturb_strenth
    return latent_n + disturb


def inversion_attack(
    generator,
    target_model,
    embed_model,
    p2f,
    target_label,
    latent_in,
    attack_step,
    optimizer,
    lr_scheduler,
    face_shape,
    img_size,
    input_is_latent,
):

    print(f'start attack label-{target_label}!')
    save_dir = 'gen_figures/'
    pbar = tqdm(range(attack_step))
    t_resize = Resize(face_shape)

    for i in pbar:
        _disturb = 0.0
        mut_stren = 0.5

        latent_n = disturb_latent(latent_in, _disturb)
        imgs_gen, _ = generator([latent_n], input_is_latent=input_is_latent)
        batch = imgs_gen.shape[0]
        # ------------------------------------------

        if (i + 1) % 100 == 0:
            file_name = f'step{i+1}.jpg'
            uts.save_image(
                imgs_gen,
                save_dir + file_name,
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )
        if (i + 1) % attack_step == 0:
            with torch.no_grad():
                face_in = resize_img_gen_shape(imgs_gen, t_resize)
                before_no, _ = target_model.forward_feature(face_in)
                predicti = target_model.forward_classifier(before_no)

                ppff = F.softmax(predicti, dim=1)
                print('\nprediction: ')
                for k in range(batch):
                    tmp = ppff[k][target_label].item()
                    print(f'predict{k}:{tmp}\n')

                idx = []
                tmp = ppff[:, target_label : target_label + 1]
                _, iii = torch.sort(tmp, dim=0, descending=True)
                for kk in range(10):
                    idx.append(iii[kk].item())

        # ------------------------------------------
        face_input = resize_img_gen_shape(imgs_gen, t_resize)

        before_norm, outputs1 = embed_model.forward_feature(face_input)
        embedding = embed_model.forward_classifier(before_norm)

        prediction = torch.abs(torch.randn([batch, tar_classes])).cuda()
        prediction[:, target_label] = 1e18
        prediction = F.normalize(prediction, dim=1)
        inverse_feature = p2f(prediction)
        mse_loss = F.mse_loss(embedding, inverse_feature)
        loss = mse_loss

        # ------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        pbar.set_description((f'label-{target_label} CE_loss: {loss.item():.7f};'))

    # stage II
    post_de(latent_in, generator, target_model, target_label, idx)


if __name__ == '__main__':
    # ---------------------------------------------------
    # parameters
    # ---------------------------------------------------
    device = 'cuda'
    n_mean_latent = 10000
    img_size = 128

    # path of the generator
    gan_ckpt_path = 'trained_models/150000.pt'

    # path of the target model
    tar_pth = 'trained_models/FaceScrub-MobileNet-Train_Acc0.9736-Val_Acc0.9613.pth'

    # path of the embedding model
    emb_pth = 'trained_models/casia-InceptionResnet-Train_Acc0.984-Val_Acc0.971.pth'

    # path of the inverse model
    p2f_dir = 'checkpoint/10_pre2feat_FM2CI_keep100_loss_3.9467.pt'

    emb_backbone = 'inception_resnetv1'
    tar_backbone = 'mobile_net'

    tar_classes = 10575
    emb_classes = 526

    input_latent = True
    init_lr = 0.02
    init_label = 0
    fina_label = 1
    step = 20
    face_shape = [160, 160]
    batch = 16  # or 8
    load_init = False

    # ---------------------------------------------------
    # load models
    g_ema = Generator(img_size, 512, 8, channel_multiplier=1)
    g_ema.load_state_dict(
        torch.load(gan_ckpt_path, map_location='cpu')['g_ema'], strict=True
    )
    g_ema.eval()
    g_ema.to(device)

    target_model = Facenet(backbone=tar_backbone, num_classes=tar_classes)
    target_model.load_state_dict(torch.load(tar_pth, map_location='cpu'), strict=True)
    target_model.to('cuda')
    target_model.eval()

    embed_model = Facenet(backbone=emb_backbone, num_classes=emb_classes)
    embed_model.load_state_dict(torch.load(emb_pth, map_location='cpu'), strict=True)
    embed_model.to('cuda')
    embed_model.eval()

    p2f = predict2feature(tar_classes, tar_classes)
    p2f.load_state_dict(torch.load(p2f_dir, map_location='cpu')['map'], strict=True)
    p2f.to('cuda')
    p2f.eval()

    with torch.no_grad():
        noise_samples = torch.randn(n_mean_latent, 512, device=device)
        latents = g_ema.style(noise_samples)
        latent_mean = latents.mean(0)
        latent_std = (
            ((latents - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        ).item()
        # ---------------------------------------------------
        latent_in = torch.zeros((batch, 512)).to(device)
        latent_in.requires_grad = True

    noises = None

    optimizer = optim.Adam([latent_in], lr=init_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    print(f'attack from {init_label} to {fina_label-1}')
    for target_label in range(init_label, fina_label):
        with torch.no_grad():
            for i in range(batch):
                j = random.randint(0, n_mean_latent // 10 - 100)
                tmp = latents[2 * j : 2 * (j + 1), :].detach().mean(0).clone()
                latent_in[i, :] = tmp

        optimizer = optim.Adam([latent_in], lr=init_lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

        inversion_attack(
            g_ema,
            target_model,
            embed_model,
            p2f,
            target_label,
            latent_in,
            step,
            optimizer,
            lr_scheduler,
            face_shape,
            img_size,
            input_latent,
        )
