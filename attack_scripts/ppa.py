import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

# from modelinversion.attack.PLGMI.attack import attack as plgmi_attack
from modelinversion.attack.PPA.attacker import PPAAttackConfig, PPAAttacker
from development_config import get_dirs
from torchvision.transforms import *
from torchvision.transforms import functional as tvf

if __name__ == '__main__':
    dirs = get_dirs('ppa')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    # target name support: vgg16, ir152, facenet64, facenet
    target_name = 'densenet169'
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'inception_v3'
    # gan target name support: vgg16
    gan_target_name = 'vgg16'
    # dataset name support: celeba
    dataset_name = 'celeba'
    
    batch_size = 5
    target_labels = [1] # list(range(1))
    device = 'cuda:0'
    
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
        stylegan_file_path='/data/qyx0409/intermediate-MIA/stylegan2-ada-pytorch/ffhq.pkl',
        init_select_transform=init_transform,
        attack_transform=attack_transform,
        to_result_transform=to_result_transform,
        final_select_transform=fianl_selection_transform
    )
    
    
    
    attacker = PPAAttacker(config)
    
    attacker.attack(batch_size, target_labels)
    
    # attacker.evaluation(64, knn=True, feature_distance=True, fid=True)