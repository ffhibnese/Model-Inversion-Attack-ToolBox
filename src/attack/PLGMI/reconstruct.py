from collections import OrderedDict
import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation

from . import losses as L
from . import utils
from ...models import *
from ...metrics import get_knn_dist, calc_fid
# from ...metrics import knn
# from ...metrics.fid.fid import calc_fid

from .models.generators.resnet64 import ResNetGenerator
from .utils import save_tensor_images
from .config import PlgmiAttackConfig
from ...utils import Tee


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(42)


def inversion(args, G, T, E, iden, itr, lr=2e-2, iter_times=1500, num_seeds=5):
    save_img_dir = os.path.join(args.save_dir, 'all_imgs')
    success_dir = os.path.join(args.save_dir, 'success_imgs')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    bs = iden.shape[0]
    iden = iden.view(-1).long().to(args.device)

    G.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed

        set_random_seed(random_seed)

        z = utils.sample_z(
            bs, args.gen_dim_z, args.device, args.gen_distribution
        )
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(iter_times):
            fake = G(z, iden)
            out1 = T(aug_list(fake)).result
            out2 = T(aug_list(fake)).result

            if z.grad is not None:
                z.grad.data.zero_()

            if args.inv_loss_type == 'ce':
                inv_loss = L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
            elif args.inv_loss_type == 'margin':
                inv_loss = L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
            elif args.inv_loss_type == 'poincare':
                inv_loss = L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake_img = G(z, iden)
                    eval_prob = E(augmentation.Resize((112, 112))(fake_img)).result
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        with torch.no_grad():
            fake = G(z, iden)
            score = T(fake).result
            eval_prob = E(augmentation.Resize((112, 112))(fake)).result
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                sample = G(z, iden)[i]
                all_img_class_path = os.path.join(save_img_dir, str(gt))
                if not os.path.exists(all_img_class_path):
                    os.makedirs(all_img_class_path)
                save_tensor_images(sample.detach(),
                                   os.path.join(all_img_class_path, "attack_iden_{}_{}.png".format(gt, r_idx)))

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = G(z, iden)[i]
                    success_img_class_path = os.path.join(success_dir, str(gt))
                    if not os.path.exists(success_img_class_path):
                        os.makedirs(success_img_class_path)
                    save_tensor_images(best_img.detach(), os.path.join(success_img_class_path,
                                                                       "{}_attack_iden_{}_{}.png".format(itr, gt,
                                                                                                         int(no[i]))))
                    no[i] += 1
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            if args.device == 'cuda':
                torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5

from dataclasses import dataclass

@dataclass
class PlgmiArgs:
    taregt_name: str
    eval_name: str
    save_dir: str
    path_G: str
    device: str
    
    inv_loss_type: str = 'margin'
    lr: float = 0.1
    iter_times: int = 600
    gen_num_features: int = 64
    gen_dim_z: int = 128
    gen_bottom_width: int = 4
    gen_distribution: str = 'normal'
    



# def _plgmi_attack(target_name, eval_name, ckpt_dir, dataset_name, dataset_dir, result_dir, batch_size=20, cgan_target_name='vgg16', device='cpu', **kwargs):
def plgmi_attack(attack_args: PlgmiAttackConfig):
    
    target_name = attack_args.target_name
    eval_name = attack_args.eval_name
    ckpt_dir = attack_args.ckpt_dir
    dataset_name = attack_args.dataset_name
    batch_size = attack_args.batch_size
    result_dir = attack_args.result_dir
    cgan_target_name = attack_args.cgan_target_name
    device = attack_args.device
    
    save_dir = os.path.join(result_dir, f'{dataset_name}_{target_name}_{cgan_target_name}')
    os.makedirs(save_dir, exist_ok=True)
    Tee(f'{save_dir}/attack.log', 'w')
    
    args = PlgmiArgs(target_name, eval_name, save_dir, ckpt_dir, device=device,
                     inv_loss_type=attack_args.inv_loss_type,
                     lr=attack_args.lr,
                     iter_times=attack_args.iter_times,
                     gen_num_features=attack_args.gen_num_features,
                     gen_dim_z=attack_args.gen_dim_z,
                     gen_bottom_width=attack_args.gen_bottom_width,
                     gen_distribution=attack_args.gen_distribution)


    print("=> creating model ...")

    set_random_seed(42)

    # load Generator
    G = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        num_classes=1000, distribution=args.gen_distribution
    )
    gen_ckpt_path = os.path.join(ckpt_dir, 'PLG_MI', f'{dataset_name}_{cgan_target_name.upper()}_PLG_MI_G.tar')

    gen_ckpt = torch.load(gen_ckpt_path) #['model']
    if isinstance(gen_ckpt, dict):
        if 'state_dict' in gen_ckpt:
            gen_ckpt = gen_ckpt['state_dict']
        elif 'model' in gen_ckpt:
            gen_ckpt = gen_ckpt['model']
    G.load_state_dict(gen_ckpt)
    G = G.to(args.device)

    # Load target model
    if args.taregt_name.startswith("vgg16"):
        T = VGG16(1000)
        path_T = os.path.join(ckpt_dir, 'celeba', 'VGG16_88.26.tar')
    elif args.taregt_name.startswith('ir152'):
        T = IR152(1000)
        path_T = os.path.join(ckpt_dir, 'celeba', 'IR152_91.16.tar')
    elif args.taregt_name == "facenet64":
        T = FaceNet64(1000)
        path_T = os.path.join(ckpt_dir, 'celeba', 'FaceNet64_88.50.tar')
    T = (T).to(args.device)
    ckp_T = torch.load(path_T)['state_dict']
    T.load_state_dict(ckp_T, strict=True)

    # Load evaluation model
    E = FaceNet(1000)
    E = (E).to(args.device)
    path_E = os.path.join(ckpt_dir, 'celeba', 'FaceNet_95.88.tar')
    ckp_E = torch.load(path_E)['state_dict']
    E.load_state_dict(ckp_E, strict=True)
    # E.load_state_dict(ckp_E['state_dict'], strict=False)

    print("=> Begin attacking ...")
    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    for i in range(1):
        # attack 60 classes per batch
        iden = torch.from_numpy(np.arange(batch_size))

        # evaluate on the first 300 identities only
        for idx in range(5):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            # reconstructed private images
            acc, acc5, var, var5 = inversion(args, G, T, E, iden, itr=i, lr=args.lr, iter_times=args.iter_times,
                                             num_seeds=5)

            iden = iden + batch_size
            aver_acc += acc / 5
            aver_acc5 += acc5 / 5
            aver_var += var / 5
            aver_var5 += var5 / 5

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                            aver_acc5,
                                                                                                            aver_var,
                                                                                                            aver_var5))

    print("=> Calculate the KNN Dist.")
    knn_dist = get_knn_dist(E, os.path.join(args.save_dir, 'all_imgs'), os.path.join(ckpt_dir, 'PLG_MI', "celeba_private_feats"), resolution=112)
    print("KNN Dist %.2f" % knn_dist)

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(args.save_dir, "success_imgs"),
                   private_img_path= os.path.join(ckpt_dir, 'PLG_MI', "datasets", "celeba_private_domain"),
                   batch_size=batch_size)
    print("FID %.2f" % fid)
