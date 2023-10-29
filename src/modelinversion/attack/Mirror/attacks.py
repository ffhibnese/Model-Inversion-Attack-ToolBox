from .mirror.presample import presample
from .mirror.blackbox.blackbox_attack import mirror_blackbox_attack, MirrorBlackBoxArgs
from .mirror.whitebox.whitebox_attack import mirror_white_box_attack, MirrorWhiteBoxArgs
import os
from .config import MirrorBlackBoxConfig
from ...utils import Tee, FolderManager
from ...models import get_model
from .genforce.get_genforce import get_genforce

# def blackbox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, dataset_name, result_dir=None, batch_size=10, device='cpu', calc_knn=False):

def blackbox_attack(config: MirrorBlackBoxConfig):
    cache_dir = config.cache_dir
    target_name = config.target_name
    eval_name = config.eval_name
    ckpt_dir = config.ckpt_dir
    genforce_name = config.genforce_name
    batch_size = config.batch_size
    device = config.device
    target_labels = config.target_labels
    
    cache_dir = os.path.join(cache_dir, 'blackbox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(config.result_dir, 'blackbox', f'{target_name}_{eval_name}')
    presample_dir = os.path.join(cache_dir, 'pre_sample', genforce_name)
    
    folder_manager = FolderManager(ckpt_dir, None, cache_dir, result_dir, presample_dir=presample_dir)
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=10000, batch_size=config.batch_size, device=config.device)
    
    target_model = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(target_model, config.dataset_name, config.target_name, device=config.device)
    
    eval_model = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(eval_model, config.dataset_name, config.eval_name, device=config.device)
    
    args = MirrorBlackBoxArgs(
        population=1000,
        arch_name=config.target_name,
        eval_name=eval_name,
        genforce_model_name=config.genforce_name,
        target_labels=target_labels,
        batch_size=batch_size,
        device=device,
        calc_knn = False
    )
    
    
    generator, _ = get_genforce(config.genforce_name, config.device, config.ckpt_dir, use_discri=False, use_w_space=args.use_w_space, use_z_plus_space=False, repeat_w=args.repeat_w)
    
    mirror_blackbox_attack(args, generator, target_model, eval_model, folder_manager=folder_manager)
    
def white_attack(config: MirrorBlackBoxConfig):
    # work_dir = config.cache_dir
    target_name = config.target_name
    eval_name = config.eval_name
    ckpt_dir = config.ckpt_dir
    genforce_name = config.genforce_name
    # result_dir = args.result_dir
    batch_size = config.batch_size
    dataset_name = config.dataset_name
    device = config.device
    target_labels = config.target_labels
    
    if batch_size % len(target_labels) != 0:
        raise RuntimeError('batch size shoube be divisioned by number of target labels')
    
    cache_dir = os.path.join(config.cache_dir, 'whitebox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(config.result_dir, 'whitebox', f'{target_name}_{eval_name}')
    presample_dir = os.path.join(cache_dir, 'pre_sample', genforce_name)
    
    folder_manager = FolderManager(ckpt_dir, None, cache_dir, result_dir, presample_dir=presample_dir)
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=10000, batch_size=config.batch_size, device=config.device)
    
    target_model = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(target_model, config.dataset_name, config.target_name, device=config.device)
    
    eval_model = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(eval_model, config.dataset_name, config.eval_name, device=config.device)
        
    calc_knn = dataset_name == 'celeba'
        
    args = MirrorWhiteBoxArgs(
        arch_name=config.target_name,
        test_arch_name = config.eval_name,
        genforce_model_name=config.genforce_name,
        target_labels=target_labels,
        device=config.device,
        calc_knn=calc_knn,
        batch_size=config.batch_size
    )
    # mirror_white_box_attack(target_name, eval_name, genforce_name, target_labels, cache_dir, ckpt_dir, ckpt_dir, result_dir ,dataset_name, presample_dir, False, batch_size=batch_size, calc_knn=calc_knn, device=device)
    
    mirror_white_box_attack(args, target_model, eval_model, folder_manager = folder_manager)