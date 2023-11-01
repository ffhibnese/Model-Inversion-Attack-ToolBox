import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

# from modelinversion.attack.DeepInversion.imagenet_inversion import deepinversion_attack
from modelinversion.attack.DeepInversion.config import DeepInversionConfig
from modelinversion.attack.DeepInversion.attack import attack as deepinversion_attack
from development_config import get_dirs

if __name__ == '__main__':
    
    dirs = get_dirs('deepinversion')
    work_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    
    # target and eval name from torchvision
    target_name = 'resnet50'
    eval_name = 'mobilenet_v2'
    target_labels = list(range(60))
    device = 'cpu'
    batch_size = 60
    
    config = DeepInversionConfig(
        target_name = target_name,
        eval_name=eval_name,
        target_labels = target_labels,
        cache_dir=work_dir,
        result_dir=result_dir,
        dataset_name='imagenet',
        device=device,
        batch_size=batch_size
    )
    
    deepinversion_attack(config)