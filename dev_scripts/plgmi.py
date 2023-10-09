import sys
sys.path.append('.')

from attack.PLGMI.reconstruct import plgmi_attack
from development_config import get_dirs

if __name__ == '__main__':
    work_dir, result_dir, ckpt_dir = get_dirs('mirror')
    
    target_name = 'vgg16'
    eval_name = 'facenet'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = [108, 180] + list(range(10))
    
    plgmi_attack(target_name, eval_name, work_dir, ckpt_dir)