from .config import BrepAttackConfig
from .code.brep import BrepArgs, brep_attack
import os
import torch
from ...utils import FolderManager
from ...models import get_model
from ..KEDMI.code.generator import Generator
from ...metrics import calc_knn, generate_private_feats

def attack(config: BrepAttackConfig):
    
    save_dir = os.path.join(config.result_dir, f'{config.dataset_name}_{config.target_name}_{config.gan_target_name}')
    folder_manager = FolderManager(config.ckpt_dir, config.dataset_dir, config.cache_dir, save_dir, config.defense_ckpt_dir, config.defense_type)
    
    args = BrepArgs(batch_dim_for_initial_points=config.batch_size)
    
    T = get_model(config.target_name, config.dataset_name, config.device)
    folder_manager.load_target_model_state_dict(T, config.dataset_name, config.target_name, device=config.device, defense_type=config.defense_type)
    
    E = get_model(config.eval_name, config.dataset_name, config.device)
    folder_manager.load_target_model_state_dict(E, config.dataset_name, config.eval_name, device=config.device)
    
    G = Generator(args.z_dim).to(config.device)
    folder_manager.load_state_dict(G, 
                                   ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_G.tar'],
                                   device=config.device)
    
    T.eval()
    E.eval()
    G.eval()
    
    # acc = brep_attack(args, G, T, E, config.target_labels, folder_manager, device=config.device)
    
    # print(f'eval acc: {acc:.6f}')
    
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