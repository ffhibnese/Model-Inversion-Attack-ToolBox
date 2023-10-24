import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.GMI.attack import attack as gmi_attack
# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
from modelinversion.attack.GMI.config import GmiAttackConfig
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('plgmi')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'facenet64'
    eval_name = 'facenet'
    gan_target_name = 'vgg16'
    dataset_name = 'facescrub'
    
    batch_size = 60
    device = 'cuda'
    
    config = GmiAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        gan_target_name=gan_target_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_name=dataset_name,
        # dataset_dir=dataset_dir,
        device=device,
        batch_size=batch_size
    )
    
    gmi_attack(config)