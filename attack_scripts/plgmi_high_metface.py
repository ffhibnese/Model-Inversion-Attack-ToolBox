import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

# from modelinversion.attack.PLGMI.attack import attack as plgmi_attack
from modelinversion.attack.PLGMI_high.attacker import PLGMIAttacker, PLGMIAttackConfig
# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
# from modelinversion.attack.PLGMI.config import PLGMIAttackConfig
from development_config import get_dirs
# 
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    dirs = get_dirs('plgmi_high_metfaces')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    # target name support: vgg16, ir152, facenet64, facenet
    target_name = 'resnet18'
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'inception_v3'
    # gan target name support: vgg16
    gan_target_name = 'resnet18'
    # dataset name support: celeba
    dataset_name = 'facescrub'
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = 'metfaces'
    
    batch_size = 70
    # target_labels = list(range(512, 544))
    target_labels = list(range(0, 530))
    device = 'cuda'
    
    config = PLGMIAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        gan_target_name=gan_target_name,
        gan_dataset_name=gan_dataset_name,
        gen_num_per_target=50
    )
    
    attacker = PLGMIAttacker(config)
    
    attacker.attack(batch_size, target_labels)
    
    # attacker.evaluation(200, knn=True, feature_distance=True, fid=True)