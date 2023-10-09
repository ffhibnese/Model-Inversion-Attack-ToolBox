import sys
sys.path.append('.')

from development_config import get_dirs
from attack import mirror_whitebox_attack


if __name__ == '__main__':
    work_dir, result_dir, ckpt_dir = get_dirs('mirror')
    
    target_name = 'vgg16'
    eval_name = 'facenet'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = [108, 180] + list(range(10))
    
    mirror_whitebox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, result_dir, batch_size=len(target_labels), device='cuda')