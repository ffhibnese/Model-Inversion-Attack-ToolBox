import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.Mirror.attacker import (
    MirrorWhiteboxAttackConfig,
    MirrorWhiteboxAttacker,
)
from development_config import get_dirs


if __name__ == '__main__':
    dirs = get_dirs('mirror')
    cache_dir, result_dir, ckpt_dir, dataset_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
    )

    target_name = 'vgg16'
    eval_name = 'facenet'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = list(range(30))
    dataset_name = 'celeba'

    batch_size = 10

    device = 'cuda:1'

    config = MirrorWhiteboxAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        genforce_name=genforce_name,
        presample_batch_size=64,
    )

    attacker = MirrorWhiteboxAttacker(config)

    attacker.attack(batch_size, target_labels=target_labels)

    attacker.evaluation(batch_size)
