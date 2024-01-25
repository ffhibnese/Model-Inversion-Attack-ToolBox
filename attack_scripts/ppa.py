import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

# from modelinversion.attack.PLGMI.attack import attack as plgmi_attack
from modelinversion.attack.PPA.attacker import PPAAttackConfig, PPAAttacker
from development_config import get_dirs
from torchvision.transforms import *
from torchvision.transforms import functional as tvf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':
    dirs = get_dirs('ppa')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'densenet169'
    eval_name = 'inception_v3'
    # dataset name support: celeba
    dataset_name = 'hdceleba'
    stylegan_dataset = 'ffhq'
    
    batch_size = 5
    target_labels = [0,1,2,3,4,5,6,7,8,9] # list(range(1))
    device = 'cuda'
    
    def init_transform(img):
        img = tvf.center_crop(img, (800, 800))
        img = tvf.resize(img, (224, 224), antialias=True)
        imgs = [img, tvf.hflip(img)]
        return imgs
    
    attack_transform = Compose([
        CenterCrop((800, 800)),
        Resize((224, 224), antialias=True),
        RandomResizedCrop((224, 224), scale=[0.9, 1.0], ratio=[1.0, 1.0], antialias=True)
    ])
    
    to_result_transform = Compose([
        CenterCrop((800, 800)),
        Resize((224, 224), antialias=True),
    ])
    
    to_eval_transform = Compose([
        CenterCrop((800, 800)),
        Resize((299, 299), antialias=True),
    ])
    
    fianl_selection_transform = Compose([
            RandomResizedCrop(size=(224, 224),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),
            RandomHorizontalFlip(0.5)
        ])
        
    
    config = PPAAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        stylegan_resp_dir='/data/qyx0409/intermediate-MIA/stylegan2-ada-pytorch',
        stylegan_file_path=f'/data/qyx0409/intermediate-MIA/stylegan2-ada-pytorch/{stylegan_dataset}.pkl',
        stylegan_dataset=stylegan_dataset,
        init_select_transform=init_transform,
        attack_transform=attack_transform,
        to_result_transform=to_result_transform,
        to_eval_transform=to_eval_transform,
        final_select_transform=fianl_selection_transform,
        num_epochs=70
    )
    
    
    
    attacker = PPAAttacker(config)
    
    attacker.attack(batch_size, target_labels)
    
    # attacker.evaluation(64, knn=True, feature_distance=True, fid=True)