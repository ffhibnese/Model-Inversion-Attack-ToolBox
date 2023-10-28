from .mirror.presample import presample
from .mirror.blackbox.blackbox_attack import mirror_blackbox_attack
from .mirror.whitebox.whitebox_attack import mirror_white_box_attack
import os
from .config import MirrorBlackBoxConfig
from ...utils import Tee

# def blackbox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, dataset_name, result_dir=None, batch_size=10, device='cpu', calc_knn=False):

def blackbox_attack(args: MirrorBlackBoxConfig):
    work_dir = args.cache_dir
    target_name = args.target_name
    eval_name = args.eval_name
    ckpt_dir = args.ckpt_dir
    genforce_name = args.genforce_name
    # result_dir = args.result_dir
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    device = args.device
    target_labels = args.target_labels
    
    cache_dir = os.path.join(work_dir, 'blackbox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(args.result_dir, 'blackbox', f'{target_name}_{eval_name}')
    # ckpt_dir = os.path.join(work_dir, 'models')
    presample_dir = os.path.join(work_dir, 'pre_sample', genforce_name)
    
    
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(presample_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    Tee(os.path.join(result_dir, 'attack.log'), 'w')
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=8000, batch_size=40)
        
    mirror_blackbox_attack(1000, target_name, eval_name, genforce_name, ckpt_dir, result_dir, presample_dir, target_labels, cache_dir, ckpt_dir, dataset_name=dataset_name, batch_size=batch_size, use_cache=False, device=device, calc_knn=False)
    
    # mirror_blackbox_attack(False, 1000, eval_name, genforce_name, ckpt_dir, result_dir, presample_dir, target_labels, cache_dir, ckpt_dir, dataset_name=dataset_name, batch_size=batch_size, use_cache=False, device=device, calc_knn=calc_knn)
    
def white_attack(args: MirrorBlackBoxConfig):
    work_dir = args.cache_dir
    target_name = args.target_name
    eval_name = args.eval_name
    ckpt_dir = args.ckpt_dir
    genforce_name = args.genforce_name
    # result_dir = args.result_dir
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    device = args.device
    target_labels = args.target_labels
    
    if batch_size % len(target_labels) != 0:
        raise RuntimeError('batch size shoube be divisioned by number of target labels')
    cache_dir = os.path.join(work_dir, 'whitebox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(args.result_dir, 'whitebox', f'{target_name}_{eval_name}')
    # ckpt_dir = os.path.join(work_dir, 'models')
    presample_dir = os.path.join(work_dir, 'pre_sample', genforce_name)
    
    Tee(os.path.join(result_dir, 'attack.log'), 'w')
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(presample_dir, exist_ok=True)
    
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=8000, batch_size=40)
        
    calc_knn = dataset_name == 'celeba'
        
    mirror_white_box_attack(target_name, eval_name, genforce_name, target_labels, cache_dir, ckpt_dir, ckpt_dir, result_dir ,dataset_name, presample_dir, False, batch_size=batch_size, calc_knn=calc_knn, device=device)