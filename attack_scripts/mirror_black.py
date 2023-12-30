import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.Mirror.attacker import MirrorBlackboxAttacker, MirrorBlackboxAttackConfig
from development_config import get_dirs


if __name__ == '__main__':
    dirs = get_dirs('mirror')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'resnet50_scratch_dag'
    eval_name = 'inception_resnetv1'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = [108, 180]
    dataset_name = 'vggface2'
    
    device = 'cuda:2'
    
    config = MirrorBlackboxAttackConfig(
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
        population=1000
    )
    
    attacker = MirrorBlackboxAttacker(config)
    
    attacker.attack(2, target_labels)