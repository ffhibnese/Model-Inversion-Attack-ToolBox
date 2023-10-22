import sys
sys.path.append('.')

from attack.PLGMI.attack import plgmi_attack
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('plgmi')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'facenet64'
    eval_name = 'facenet'
    dataset_name = 'celeba'
    cgan_target_name = 'ir152'
    
    
    plgmi_attack(target_name, eval_name, ckpt_dir, dataset_name, dataset_dir, result_dir, cgan_target_name=cgan_target_name)