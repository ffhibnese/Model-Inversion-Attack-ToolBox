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
from utils.DE_mask import Optimizer as post_opt
from utils.DE_mask import DE_c2b_5_bin, DE_c2b_5_bin2


# resize生成的图像为指定形状
def resize_img_gen_shape(img_gen, trans):
    t_img = trans(img_gen)
    face_input = t_img.clamp(min=-1, max=1).add(1).div(2)
    return face_input


# 第二阶段的gradient-free攻击
def post_de(latent_in, generator, target_model, target_label, idx):

    x = latent_in[idx]
    optim_DE = post_opt(
        target_model,
        generator,
        target_label,
        trunc,
        direct=f'gen_figures/DE_facescrub_mobile_M{trunc}_counter/',
    )
    # task = DE_c2b_5_bin(optim_DE, max_gen=300, x=x)
    task = DE_c2b_5_bin2(optim_DE, max_gen=250, x=x)
    task.run(disturb=0.00)
    task.get_img(32, only_best=True)


# 为隐向量施加扰动
def disturb_latent(latent_in, disturb_strenth):
    latent_n = latent_in
    disturb = torch.randn_like(latent_n) * disturb_strenth
    return latent_n + disturb


# 反演攻击
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

    # 分别攻击某一种类别
    print(f'start attack label-{target_label}!')
    save_dir = 'gen_figures/'  # 保存路径
    pbar = tqdm(range(attack_step))
    t_resize = Resize(face_shape)  # resize图片为160×160

    # 第1阶段训练
    for i in pbar:
        _disturb = 0.02 * (1 - i / attack_step)  # 扰动因子
        mut_stren = 0.5

        # 根据扰动后的隐向量，生成对应图片
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

        # 完成50轮迭代后，获得第一阶段的粗粒度隐向量结果
        if (i + 1) % attack_step == 0:
            with torch.no_grad():
                # 用target测试粗粒度隐向量的效果
                face_in = resize_img_gen_shape(imgs_gen, t_resize)
                before_no, _ = target_model.forward_feature(face_in)
                predicti = target_model.forward_classifier(before_no)

                # 展示正确标签的预测置信度
                ppff = F.softmax(predicti, dim=1)
                print('\nprediction: ')
                for k in range(batch):
                    tmp = ppff[k][target_label].item()
                    print(f'predict{k}:{tmp}\n')

                # 获取当前batch各个隐向量的索引
                idx = list(range(batch))

        # ------------------------------------------
        face_input = resize_img_gen_shape(imgs_gen, t_resize)

        # 特征提取
        before_norm, outputs1 = embed_model.forward_feature(face_input)
        embedding = embed_model.forward_classifier(before_norm)

        # 获得正确的预测标签
        prediction = torch.abs(torch.randn([batch, tar_classes])).cuda()
        prediction[:, target_label] = 1e18
        prediction = F.normalize(prediction, dim=1)

        # 根据inverse network获得映射到的特征，从而计算L2损失
        inverse_feature = p2f(prediction)
        mse_loss = F.mse_loss(embedding, inverse_feature)
        loss = mse_loss

        # ------------------------------------------
        # 然后对隐向量进行优化
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
    trunc = 100  # mask

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

    tar_classes = 526  # 总的目标类数
    emb_classes = 10575

    input_latent = True
    init_lr = 0.02  # 初始学习率
    init_label = 0  # 攻击的起始标签
    fina_label = 1  # 攻击的截止标签
    step = 50  # 第1阶段迭代次数
    face_shape = [160, 160]  # 图片分辨率
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

    # inverse network
    p2f = predict2feature(tar_classes, trunc)
    p2f.load_state_dict(torch.load(p2f_dir, map_location='cpu')['map'], strict=True)
    p2f.to('cuda')
    p2f.eval()

    # 获得待优化的初始隐向量
    with torch.no_grad():
        noise_samples = torch.randn(n_mean_latent, 512, device=device)
        latents = g_ema.style(noise_samples)
        latent_mean = latents.mean(0)
        latent_std = (
            ((latents - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        ).item()
        latent_in = torch.zeros((batch, 512)).to(device)
        latent_in.requires_grad = True

    noises = None
    optimizer = optim.Adam([latent_in], lr=init_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    print(f'attack from {init_label} to {fina_label-1}')
    for target_label in range(init_label, fina_label):
        # 为当前batch的每一个隐向量进行初始化
        with torch.no_grad():
            for i in range(batch):
                j = random.randint(0, n_mean_latent // 10 - 100)
                tmp = latents[2 * j : 2 * (j + 1), :].detach().mean(0).clone()
                latent_in[i, :] = tmp

        optimizer = optim.Adam([latent_in], lr=init_lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

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
