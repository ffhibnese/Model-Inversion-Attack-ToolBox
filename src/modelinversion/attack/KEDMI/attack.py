
from .config import KedmiAttackConfig

import os
from .code.recovery import dist_inversion
from .code.generator import Generator
from .code.discri import DGWGAN, MinibatchDiscriminator
import torch
from ...models import *
from ...utils import FolderManager
import numpy as np
from ...metrics import calc_knn, generate_private_feats, calc_fid

def attack(config: KedmiAttackConfig):
    
    save_dir = os.path.join(config.result_dir, f'{config.gan_dataset_name}_{config.target_name}')
    folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, save_dir)

    print("=> creating model ...")


    z_dim = 100
    
    G = Generator(z_dim).to(config.device)
    D = MinibatchDiscriminator().to(config.device)
    
    folder_manager.load_state_dict(G, 
                                   ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_G.tar'],
                                   device=config.device)
    folder_manager.load_state_dict(D, 
                                   ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_D.tar'],
                                   device=config.device)

    T = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=config.device)

    E = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(E, config.dataset_name, config.eval_name, device=config.device)

    ############         attack     ###########
    print("=> Begin attacking ...")

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    
    if len(config.target_labels) > 0:
    
        # evaluate on the first 300 identities only
        for idx in range((len(config.target_labels) - 1) // config.batch_size + 1):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            
            iden = torch.tensor(
                config.target_labels[idx * config.batch_size: min((idx+1)*config.batch_size, len(config.target_labels))], device=config.device, dtype=torch.long
                )

            acc, acc5, var, var5 = dist_inversion(G, D, T, E, iden, folder_manager=folder_manager, lr=2e-2, momentum=0.9, lamda=100,
                                                iter_times=1500, clip_range=1, 
                                                    device=config.device)

            aver_acc += acc * len(iden) / len(config.target_labels)
            aver_acc5 += acc5 * len(iden) / len(config.target_labels)
            aver_var += var * len(iden) / len(config.target_labels)
            aver_var5 += var5 * len(iden) / len(config.target_labels)

        print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                                aver_acc5,
                                                                                                                aver_var,
                                                                                                                aver_var5))

    
    # print("=> Calculate the KNN Dist.")
    # knn_dist = get_knn_dist(E, os.path.join(save_dir, 'all_imgs'), os.path.join(config.ckpt_dir, 'PLGMI', "celeba_private_feats"), resolution=112, device=config.device)
    # print("KNN Dist %.2f" % knn_dist)
    
    print("=> Calculate the KNN Dist.")
    
    
    generate_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, config.target_name)
    private_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, 'private')
    
    if config.dataset_name == 'celeba':
        private_img_dir = os.path.join(config.dataset_dir, config.dataset_name, 'split', 'private', 'train')
    else:
        raise NotImplementedError(f'dataset {config.dataset_name} is NOT supported')
    
    generate_private_feats(eval_model=E, img_dir=os.path.join(save_dir, 'all_imgs'), save_dir=generate_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None)
    generate_private_feats(eval_model=E, img_dir=private_img_dir, save_dir=private_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None, exist_ignore=True)
    
    knn_dist = calc_knn(generate_feat_save_dir, private_feat_save_dir)
    print("KNN Dist %.2f" % knn_dist)
    
    # print("=> Calculate the FID.")
    # fid = calc_fid(recovery_img_path=os.path.join(save_dir, "success_imgs"),
    #                private_img_path= os.path.join(ckpt_dir, 'PLGMI', "datasets", "celeba_private_domain"),
    #                batch_size=batch_size)
    # print("FID %.2f" % fid)

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(save_dir, "all_imgs"),
                   private_img_path= os.path.join(config.dataset_dir, config.dataset_name, "split", "private", "train"),
                   batch_size=config.batch_size, device=config.device)
    print("FID %.2f" % fid)