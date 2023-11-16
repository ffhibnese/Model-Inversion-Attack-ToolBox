import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.KEDMI.attack import attack as kedmi_attack
from modelinversion.attack.KEDMI.config import KedmiAttackConfig
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('vib_kedmi')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    target_name = 'vgg16'
    eval_name = 'facenet'
    gan_target_name = 'vgg16'
    dataset_name = 'celeba'

    gan_dataset_name = 'celeba'
    
    batch_size = 60
    target_labels = list(range(120))
    device = 'cuda:0'
    
    config = KedmiAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        gan_target_name=gan_target_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        gan_dataset_name=gan_dataset_name,
        target_labels=target_labels,
        device=device,
        batch_size=batch_size,
        defense_type='vib',
        defense_ckpt_dir=defense_ckpt_dir
    )
    
    kedmi_attack(config)