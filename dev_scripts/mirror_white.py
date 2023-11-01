import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.attack.Mirror.attacks import white_attack
# from modelinversion.attack.PLGMI.reconstruct import plgmi_attack
from modelinversion.attack.Mirror.config import MirrorWhiteBoxConfig
from development_config import get_dirs


if __name__ == '__main__':
    dirs = get_dirs('mirror')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    # target_name = 'resnet50_scratch_dag'
    # eval_name = 'inception_resnetv1'
    # genforce_name = 'stylegan_celeba_partial256'
    # target_labels = [108, 180] + list(range(18))
    # dataset_name = 'vggface2'
    
    target_name = 'vgg16'
    eval_name = 'facenet'
    genforce_name = 'stylegan_celeba_partial256'
    target_labels = list(range(71, 71+300))
    dataset_name = 'celeba'
    
    batch_size = 30
    
    device = 'cuda:0'
    
    config = MirrorWhiteBoxConfig(
        target_name=target_name,
        eval_name=eval_name,
        genforce_name=genforce_name,
        ckpt_dir=ckpt_dir,
        cache_dir=work_dir,
        result_dir=result_dir,
        dataset_dir = dataset_dir,
        dataset_name=dataset_name,
        target_labels=target_labels,
        batch_size=batch_size,
        device=device
    )
    
    white_attack(config)
    # mirror_whitebox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, dataset_name, result_dir, batch_size=len(target_labels), device='cuda', calc_knn=calc_knn)