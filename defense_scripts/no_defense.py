import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, ToPILImage, Lambda, Resize

from development_config import get_dirs
from modelinversion.defense import *
from modelinversion.models import get_model
from modelinversion.foldermanager import FolderManager
from modelinversion.utils import RandomIdentitySampler

if __name__ == '__main__':
    
    defense_type = 'no_defense'
    
    dirs = get_dirs(defense_type)
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type)
    
    model_name = 'efficientnet_b0'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.01
    device = 'cuda:0'
    batch_size = 64
    
    args = BaseTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='no_defense',
        tqdm_strategy=TqdmStrategy.ITER,
        device=device
    )
    
    model = get_model(model_name, dataset_name, device, backbone_pretrain=True)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    trainer = RegTrainer(args, folder_manager, model, optimizer, None)
    
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=Compose([
        RandomHorizontalFlip(p=0.5), ToTensor()
    ]))
    # print(trainset[0][0].shape)
    train_sampler = RandomIdentitySampler(trainset, batch_size, 4)
    trainloader = DataLoader(trainset, batch_size, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    trainer.train(trainloader, testloader)
