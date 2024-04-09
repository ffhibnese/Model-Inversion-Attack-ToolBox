import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from modelinversion.attack.Lokt.gan_trainer import LoktGANTrainArgs, LoktGANTrainer
from development_config import get_dirs

from modelinversion.foldermanager import FolderManager

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    batch_size = 64

    target_name = 'vgg16'
    dataset_name = 'celeba'
    device = 'cuda'

    dirs = get_dirs('lokt_gan')
    cache_dir, result_dir, ckpt_dir, dataset_dir = (
        dirs['work_dir'],
        dirs['result_dir'],
        dirs['ckpt_dir'],
        dirs['dataset_dir'],
    )
    dataset = ImageFolder(dataset_dir, transform=ToTensor())
    epoch_num = 300

    # trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_args = LoktGANTrainArgs(
        dataset_name=dataset_name,
        batch_size=batch_size,
        epoch_num=epoch_num,
        device=device,
        target_name=target_name,
        class_loss_start_iter=15000,
    )

    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, None)

    trainer = LoktGANTrainer(train_args, folder_manager)

    trainer.train()
    # trainer.prepare_training()
    # trainer.G.load_state_dict(torch.load('checkpoints/lokt/lokt_celeba_vgg16_G.pt')['state_dict'])
    z = torch.randn((8, 128), device=device)
    c = torch.arange(0, 8, device=device, dtype=torch.long)
    img = trainer.G(z, c)
    import torchvision

    torchvision.utils.save_image(img, 'aa.png', normalize=True, nrow=4)
