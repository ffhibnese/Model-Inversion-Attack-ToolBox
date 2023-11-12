from .reconstruct import inversion, PlgmiArgs
import os
from .config import PlgmiAttackConfig
from ...utils import FolderManager, set_random_seed
from .models.generators.resnet64 import ResNetGenerator
from ...models import *
import torch
from ...metrics.knn import generate_private_feats, calc_knn
from ...metrics.fid.fid import calc_fid

def attack(config: PlgmiAttackConfig):
    
    save_dir = os.path.join(config.result_dir, f'{config.dataset_name}_{config.target_name}_{config.cgan_target_name}')
    folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, save_dir, config.defense_ckpt_dir, config.defense_type)
    
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

    T = get_model(config.target_name, config.dataset_name, device=config.device, defense_type=config.defense_type)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=config.device, defense_type=config.defense_type)

    E = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(E, config.dataset_name, config.eval_name, device=config.device)

    print("=> Begin attacking ...")
    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0

    # evaluate on the first 300 identities only
    
    if len(config.target_labels) > 0:
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
    
    
    generate_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, config.target_name)
    private_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, 'private')
    
    if config.dataset_name == 'celeba':
        private_img_dir = os.path.join(config.dataset_dir, config.dataset_name, 'split', 'private', 'train')
    else:
        print(f'dataset {config.dataset_name} is NOT supported for KNN and FID')
        return
    
    generate_private_feats(eval_model=E, img_dir=os.path.join(save_dir, 'all_imgs'), save_dir=generate_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None)
    generate_private_feats(eval_model=E, img_dir=private_img_dir, save_dir=private_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None, exist_ignore=True)
    
    knn_dist = calc_knn(generate_feat_save_dir, private_feat_save_dir)
    print("KNN Dist %.2f" % knn_dist)
    
    

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(args.save_dir, "all_imgs"),
                   private_img_path= os.path.join(config.dataset_dir, config.dataset_name, "split", "private", "train"),
                   batch_size=config.batch_size, device=config.device)
    print("FID %.2f" % fid)