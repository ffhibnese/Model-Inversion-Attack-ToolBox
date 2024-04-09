import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.Lomma.attacker import (
    LommaKEDMIAttackConfig,
    LommaKEDMIAttacker,
)
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('lomma_kedmi')
    cache_dir, result_dir, ckpt_dir, dataset_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
    )

    # target name support: vgg16, ir152, facenet64
    target_name = 'vgg16'
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'facenet'
    # dataset name support: celeba
    dataset_name = 'celeba'
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = 'celeba'
    # augment model pretrained dataset name support: celeba, ffhq
    aug_model_dataset_name = 'celeba'

    gan_target_name: str = 'vgg16'

    batch_size = 60
    target_labels = list(range(300))
    device = 'cuda:0'

    config = LommaKEDMIAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        gan_dataset_name=gan_dataset_name,
        aug_model_dataset_name=aug_model_dataset_name,
        preg_generate_batch_size=batch_size,
        iter_times=2400,
    )

    attacker = LommaKEDMIAttacker(config)

    attacker.attack(batch_size, target_labels)

    attacker.evaluation(batch_size, knn=True, feature_distance=True, fid=True)
