import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.GMI.attacker import GMIAttackConfig, GMIAttacker
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('gmi')
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
    # dataset name support: celeba
    dataset_name = 'celeba'
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = 'celeba'

    batch_size = 60
    target_labels = list(range(120))
    device = 'cuda:2'

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
    )

    attacker = GMIAttacker(config)

    attacker.attack(batch_size, target_labels)

    attacker.evaluation(batch_size, knn=True, feature_distance=True, fid=True)
