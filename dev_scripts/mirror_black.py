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
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    target_name = 'resnet50_scratch_dag'
    eval_name = 'inception_resnetv1'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = [108, 180] + list(range(18))
    dataset_name = 'vggface2'
    
    device = 'cuda:2'
    
    config = MirrorBlackBoxConfig(
        target_name=target_name,
        eval_name=eval_name,
        genforce_name=genforce_name,
        ckpt_dir=ckpt_dir,
        cache_dir=work_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        target_labels=target_labels,
        batch_size=len(target_labels),
        device=device
    )
    
    blackbox_attack(config)