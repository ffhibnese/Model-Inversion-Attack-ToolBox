import torch
import dlib
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Resize
from torchvision import utils as uts
from torch.nn import functional as F
from .utils.DE_mask import Optimizer as post_opt
from .utils.DE_mask import DE_c2b_5_bin2
from .utils.detect_crop_face import facenet_input_preprocessing, detect_crop_face


# resize生成的图像为指定形状
def resize_img_gen_shape(img_gen, trans):
    t_img = trans(img_gen)
    face_input = t_img.clamp(min=-1, max=1).add(1).div(2)
    return face_input

# 第二阶段的gradient-free攻击
def post_de(latent_in, generator, target_model, target_label, idx, trunc, only_best, face_shape, device):

    x = latent_in[idx]
    
    # 获得差分进化（DE）的优化器
    optim_DE = post_opt(target_model, generator, target_label, trunc, 
                        face_shape=face_shape, device=device, direct=f'gen_figures/DE_facescrub_mobile_M{trunc}_counter/')
    
    # 执行差分进化
    # task = DE_c2b_5_bin(optim_DE, max_gen=300, x=x)
    task = DE_c2b_5_bin2(optim_DE, max_gen=250, x=x)
    task.run(disturb=0.00)
    return task.get_img(32, only_best=only_best)
    

# 为隐向量施加扰动
def disturb_latent(latent_in, disturb_strenth):
    latent_n = latent_in
    disturb = torch.randn_like(latent_n) * disturb_strenth
    return latent_n + disturb

# 反演攻击
def inversion_attack(generator, target_model, embed_model, p2f, target_label, latent_in, attack_step, optimizer,
                     lr_scheduler, face_shape, img_size, input_is_latent, tar_classes, trunc, device, only_best):

    # 分别攻击某一种类别
    print(f'start attack label-{target_label}!')
    save_dir   = 'gen_figures/'     # 保存路径
    pbar = tqdm(range(attack_step))
    t_resize = Resize(face_shape)   # resize图片为160×160

    # 第1阶段训练
    for i in pbar:
        _disturb  = 0.02 * (1 - i / attack_step) #扰动因子
        mut_stren = 0.5

        # 根据扰动后的隐向量，生成对应图片
        latent_n = disturb_latent(latent_in, _disturb)
        imgs_gen, _ = generator([latent_n], input_is_latent=input_is_latent)
        batch = imgs_gen.shape[0]
        # ------------------------------------------

        if (i+1) % 100 == 0:
            file_name = f'step{i+1}.jpg'
            uts.save_image(
                imgs_gen,
                save_dir+file_name,
                nrow=4,
                normalize=True,
                range=(-1, 1),
                )
            
        # 完成50轮迭代后，获得第一阶段的粗粒度隐向量结果
        if (i+1) % attack_step == 0:
            with torch.no_grad():
                # 用target测试粗粒度隐向量的效果
                face_in = resize_img_gen_shape(imgs_gen, t_resize)
                before_no, _ = target_model.forward_feature(face_in)
                predicti = target_model.forward_classifier(before_no)

                # 展示正确标签的预测置信度
                ppff   = F.softmax(predicti, dim=1)
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
        prediction = torch.abs(torch.randn([batch,tar_classes])).cuda()
        prediction[:,target_label] = 1e18
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

        pbar.set_description(
            (
                f'label-{target_label} CE_loss: {loss.item():.7f};'
            )
        )

    # stage II
    return post_de(latent_in, generator, target_model, target_label, idx, trunc, only_best, device=device, face_shape=face_shape)

def eval_acc(E, target_labels, imgs, face_shape, device):
    acc_number = 0
    top5_acc_number = 0
    conf_sum   = 0
    detector   = dlib.get_frontal_face_detector()
    for i, label in enumerate(target_labels):
        label_ = label
        _, cropped = detect_crop_face(imgs[i], detector)
        face_input = facenet_input_preprocessing(cropped, face_shape).to(device)
        
        # 获得target model的预测结果
        before_norm, outputs1 = E.forward_feature(face_input)
        prediction = E.forward_classifier(before_norm)
        conf = F.softmax(prediction, dim=1)
        out  = torch.max(conf, dim=1)
        _, rank_ = conf[0].sort(descending=True)
        rank = rank_.to(device)
        
        if label_ == out[1].item():
            acc_number += 1
        if (label_ == np.array(rank[0:5])).sum() > 0:
            top5_acc_number += 1
        conf_sum += conf[0,label_].item()
    
    acc      = acc_number / len(target_labels)
    top5_acc = top5_acc_number / len(target_labels)
    conf_avg = conf_sum / len(target_labels)
    
    return acc, top5_acc, conf_avg