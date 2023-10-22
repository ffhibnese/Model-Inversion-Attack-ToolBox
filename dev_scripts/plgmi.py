import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.PLGMI.attack import attack as plgmi_attack
# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
from modelinversion.attack.PLGMI.config import PlgmiAttackConfig
from development_config import get_dirs

# import os

# work_root_dir = './cache'
# result_root_dir = './results'
# ckpt_dir = './checkpoints'
# dataset_dir = './dataset'

# def get_dirs(method_name):
#     work_dir = os.path.join(work_root_dir, method_name)
#     result_dir = os.path.join(result_root_dir, method_name)
    
#     os.makedirs(work_dir, exist_ok=True)
#     os.makedirs(work_dir, exist_ok=True)
    
#     return {'work_dir': work_dir, 'result_dir': result_dir, 'ckpt_dir': ckpt_dir, 'dataset_dir': dataset_dir}

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
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_name=dataset_name,
        # dataset_dir=dataset_dir,
        device=device,
        batch_size=batch_size
    )
    
    plgmi_attack(config)