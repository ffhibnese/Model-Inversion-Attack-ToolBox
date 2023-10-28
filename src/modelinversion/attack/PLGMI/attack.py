from .reconstruct import inversion, PlgmiArgs
import os
from .config import PlgmiAttackConfig
from ...utils import FolderManager, set_random_seed
from .models.generators.resnet64 import ResNetGenerator
from ...models import *
import torch
from ...metrics import get_knn_dist

def attack(config: PlgmiAttackConfig):
#     plgmi_attack(args)
    
# def plgmi_attack(attack_args: PlgmiAttackConfig):
    
    # target_name = attack_args.target_name
    # eval_name = attack_args.eval_name
    # ckpt_dir = attack_args.ckpt_dir
    # dataset_name = attack_args.dataset_name
    # batch_size = attack_args.batch_size
    # result_dir = attack_args.result_dir
    # cgan_target_name = attack_args.cgan_target_name
    # device = attack_args.device
    
    save_dir = os.path.join(config.result_dir, f'{config.dataset_name}_{config.target_name}_{config.cgan_target_name}')
    folder_manager = FolderManager(config.ckpt_dir, None, None, save_dir)
    
    args = PlgmiArgs(config.target_name, config.eval_name, save_dir, config.ckpt_dir, device=config.device,
                     inv_loss_type=config.inv_loss_type,
                     lr=config.lr,
                     iter_times=config.iter_times,
                     gen_distribution=config.gen_distribution)


    print("=> creating model ...")

    set_random_seed(42)

    # load Generator
    G = ResNetGenerator(
        num_classes=1000, distribution=args.gen_distribution
    ).to(config.device)
    folder_manager.load_state_dict(G, ['PLGMI', f'{config.gan_dataset_name}_{config.cgan_target_name.upper()}_PLG_MI_G.tar'], device=args.device)

    T = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=config.device)

    E = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(E, config.dataset_name, config.eval_name, device=config.device)

    print("=> Begin attacking ...")
    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0

    # evaluate on the first 300 identities only
    for idx in range((len(config.target_labels) - 1) // config.batch_size + 1):
        print("--------------------- Attack batch [%s]------------------------------" % idx)
        
        iden = torch.tensor(
            config.target_labels[idx * config.batch_size: min((idx+1)*config.batch_size, len(config.target_labels))], device=config.device, dtype=torch.long
            )
        # reconstructed private images
        acc, acc5, var, var5 = inversion(args, G, T, E, iden, folder_manager=folder_manager, lr=config.lr, iter_times=config.iter_times,
                                            num_seeds=5)

        aver_acc += acc * len(iden) / len(config.target_labels)
        aver_acc5 += acc5  * len(iden) / len(config.target_labels)
        aver_var += var  * len(iden) / len(config.target_labels)
        aver_var5 += var5  * len(iden) / len(config.target_labels)

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                            aver_acc5,
                                                                                                            aver_var,
                                                                                                            aver_var5))

    print("=> Calculate the KNN Dist.")
    knn_dist = get_knn_dist(E, os.path.join(args.save_dir, 'all_imgs'), os.path.join(config.ckpt_dir, 'PLGMI', "celeba_private_feats"), resolution=112, device=args.device)
    print("KNN Dist %.2f" % knn_dist)

    # print("=> Calculate the FID.")
    # fid = calc_fid(recovery_img_path=os.path.join(args.save_dir, "success_imgs"),
    #                private_img_path= os.path.join(ckpt_dir, 'PLGMI', "datasets", "celeba_private_domain"),
    #                batch_size=batch_size)
    # print("FID %.2f" % fid)
