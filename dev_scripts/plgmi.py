import sys
sys.path.append('..')
sys.path.append('../src')

# from attack.PLGMI.attack import attack as plgmi_attack
from attack.PLGMI.reconstruct import plgmi_attack
from attack.PLGMI.config import PlgmiAttackConfig
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('plgmi')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'ir152'
    eval_name = 'facenet'
    dataset_name = 'celeba'
    cgan_target_name = 'ir152'
    
    batch_size = 20
    device = 'cuda'
    
    config = PlgmiAttackConfig(
        target_name=target_name,
        cgan_target_name=cgan_target_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        device=device,
        batch_size=batch_size
    )
    
    plgmi_attack(config)