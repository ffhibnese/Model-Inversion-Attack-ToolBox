import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

# from modelinversion.attack.PLGMI.attack import attack as plgmi_attack
from modelinversion.attack.PLGMI.attacker import PLGMIAttacker

# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
from modelinversion.attack.PLGMI.config import PLGMIAttackConfig
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('plgmi')
    cache_dir, result_dir, ckpt_dir, dataset_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
    )

    # target name support: vgg16, ir152, facenet64, facenet
    target_name = 'vgg16'
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'facenet'
    # gan target name support: vgg16
    gan_target_name = 'vgg16'
    # dataset name support: celeba
    dataset_name = 'celeba'
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = 'celeba'

    batch_size = 10
    target_labels = list(range(0, 10))
    device = 'cuda:2'

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
    )

    attacker = PLGMIAttacker(config)

    attacker.attack(batch_size, target_labels)

    attacker.evaluation(200, knn=True, feature_distance=True, fid=True)
