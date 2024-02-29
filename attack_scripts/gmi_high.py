import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.GMI_high.attacker import GMIAttackConfig, GMIAttacker
from development_config import get_dirs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

if __name__ == '__main__':
    dirs = get_dirs('gmi_high_rawtrain')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    # target name support: vgg16, ir152, facenet64, facenet
    target_name = 'resnet18'
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'inception_v3'
    # dataset name support: celeba
    dataset_name = 'facescrub'
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = 'ffhq256'
    
    batch_size = 60
    target_labels = list(range(60))
    device = 'cuda'
    
    config = GMIAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        gan_dataset_name=gan_dataset_name,
        gen_num_per_target=50
    )
    
    attacker = GMIAttacker(config)
    
    attacker.attack(batch_size, target_labels)
    
    # attacker.evaluation(batch_size, knn=True, feature_distance=True, fid=True)