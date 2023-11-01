# import sys
# sys.path.append('.')

# from attack import mirror_blackbox_attack
# from development_config import get_dirs

# if __name__ == '__main__':
#     dirs = get_dirs('mirror')
#     work_dir, result_dir, ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir']
    
#     target_name = 'ir152'
#     eval_name = 'facenet'
#     genforce_name = 'stylegan_celeba_partial256'
#     target_labels = [108, 180] + list(range(10))
#     dataset_name = 'celeba'
    
#     calc_knn = eval_name == 'facenet'
    
#     mirror_blackbox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, dataset_name, result_dir, batch_size=len(target_labels), device='cuda', calc_knn=calc_knn)

import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.Mirror.attacks import blackbox_attack
# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
from modelinversion.attack.Mirror.config import MirrorBlackBoxConfig
from development_config import get_dirs


if __name__ == '__main__':
    dirs = get_dirs('mirror')
    work_dir, result_dir, ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir']
    
    target_name = 'resnet50_scratch_dag'
    eval_name = 'inception_resnetv1'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = [108, 180] #+ list(range(18))
    dataset_name = 'vggface2'
    
    device = 'cuda:2'
    
    config = MirrorBlackBoxConfig(
        target_name=target_name,
        eval_name=eval_name,
        genforce_name=genforce_name,
        ckpt_dir=ckpt_dir,
        cache_dir=work_dir,
        result_dir=result_dir,
        dataset_name=dataset_name,
        target_labels=target_labels,
        batch_size=len(target_labels),
        device=device
    )
    
    blackbox_attack(config)