# from .code.recovery import gmi_attack
from .config import GmiAttackConfig
import os
from .code.recovery import inversion
from .code.generator import Generator
from .code.discri import DGWGAN
import torch
from ...models import *
from ...utils import FolderManager
import numpy as np
from ...metrics import get_knn_dist


def attack(config: GmiAttackConfig):
    
    assert config.eval_name == 'facenet'
    
    save_dir = os.path.join(config.result_dir, f'{config.gan_dataset_name}_{config.target_name}')
    folder_manager = FolderManager(config.ckpt_dir, None, None, save_dir)
    
    # save_dir = os.path.join(result_dir, f'{dataset_name}_{target_name}')
    # os.makedirs(save_dir, exist_ok=True)
    # Tee(f'{save_dir}/attack.log', 'w')

    print("=> creating model ...")


    z_dim = 100
    
    G = Generator(z_dim)
    D = DGWGAN(3)
    
    folder_manager.load_state_dict(G, 
                                   ['GMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_GMI_G.tar'],
                                   device=config.device)
    folder_manager.load_state_dict(D, 
                                   ['GMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_GMI_D.tar'],
                                   device=config.device)

    if config.target_name == "vgg16":
        T = VGG16(1000)
    elif config.target_name == 'ir152':
        T = IR152(1000)
    elif config.target_name == "facenet64":
        T = FaceNet64(1000)
    else:
        raise RuntimeError('Target model not exist')
    T = (T).to(config.device)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=config.device)

    # Load evaluation model
    E = FaceNet(1000).to(config.device)
    folder_manager.load_target_model_state_dict(E, config.dataset_name, config.eval_name, device=config.device)

    ############         attack     ###########
    print("=> Begin attacking ...")

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0

    # evaluate on the first 300 identities only
    for idx in range((len(config.target_labels) - 1) // config.batch_size + 1):
        print("--------------------- Attack batch [%s]------------------------------" % idx)
        
        iden = torch.tensor(
            config.target_labels[idx * config.batch_size: min((idx+1)*config.batch_size, len(config.target_labels))], device=config.device, dtype=torch.long
            )

        acc, acc5, var, var5 = inversion(G, D, T, E, iden, folder_manager=folder_manager, lr=2e-2, momentum=0.9, lamda=100,
                                            iter_times=1500, clip_range=1, 
                                                device=config.device)

        # iden = iden + config.batch_size
        aver_acc += acc / 5
        aver_acc5 += acc5 / 5
        aver_var += var / 5
        aver_var5 += var5 / 5

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                            aver_acc5,
                                                                                                            aver_var,
                                                                                                            aver_var5))

    
    print("=> Calculate the KNN Dist.")
    knn_dist = get_knn_dist(E, os.path.join(save_dir, 'all_imgs'), os.path.join(config.ckpt_dir, 'PLGMI', "celeba_private_feats"), resolution=112, device=config.device)
    print("KNN Dist %.2f" % knn_dist)
    
    # print("=> Calculate the FID.")
    # fid = calc_fid(recovery_img_path=os.path.join(save_dir, "success_imgs"),
    #                private_img_path= os.path.join(ckpt_dir, 'PLGMI', "datasets", "celeba_private_domain"),
    #                batch_size=batch_size)
    # print("FID %.2f" % fid)