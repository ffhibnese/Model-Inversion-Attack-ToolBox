import sys
sys.path.append('.')

from attack.PLGMI.m_cgan import plgmi_train_cgan
from development_config import get_dirs

if __name__ == '__main__':
    dirs = get_dirs('plgmi')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'ir152'
    eval_name = 'facenet'
    dataset_name = 'celeba'
    
    inv_loss_type='margin'
    
    # top_n_selection(target_name, dataset_name, ckpt_dir, dataset_dir, work_dir, device='cpu')
    plgmi_train_cgan(target_name, dataset_name, dataset_dir, work_dir, ckpt_dir, inv_loss_type=inv_loss_type)
    
    # plgmi_attack(target_name, eval_name, work_dir, ckpt_dir, dataset_name, dataset_dir)