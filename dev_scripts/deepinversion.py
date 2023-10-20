import sys
sys.path.append('.')

from attack.DeepInversion.imagenet_inversion import deepinversion_attack
from development_config import get_dirs

if __name__ == '__main__':
    
    dirs = get_dirs('deepinversion')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'resnet50'
    eval_name = 'mobilenet_v2'
    taregt_labels = list(range(60))
    batch_size = 64
    
    deepinversion_attack(work_dir, target_name, eval_name, taregt_labels, batch_size)