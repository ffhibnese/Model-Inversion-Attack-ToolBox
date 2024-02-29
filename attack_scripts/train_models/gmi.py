import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose
from modelinversion.attack.GMI.gan_trainer import GmiGANTrainArgs, GmiGANTrainer
from development_config import get_dirs

from modelinversion.foldermanager import FolderManager
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
if __name__ == '__main__':
    batch_size = 64
    
    dataset_name = 'celeba'
    target_name = 'vgg16'
    target_dataset_name = 'celeba'
    device = 'cuda'
    
    dirs = get_dirs('gmi_gan')
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']
    # dataset = ImageFolder(dataset_dir, transform=Compose([
    #     ToTensor(),
    #     CenterCrop((800,800)),
    #     Resize((256, 256), antialias=True)
    # ]))
    epoch_num = 50
    
    # trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_args = GmiGANTrainArgs(
        dataset_name=dataset_name,
        batch_size=batch_size,
        epoch_num=epoch_num,
        device=device
        # target_name=target_name,
        # target_dataset_name=target_dataset_name
    )
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, None)
    
    trainer = GmiGANTrainer(train_args, folder_manager)
    
    trainer.train()