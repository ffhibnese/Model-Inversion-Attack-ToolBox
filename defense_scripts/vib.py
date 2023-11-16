import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.defense import *
from modelinversion.utils import FolderManager
from modelinversion.models import get_model
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from development_config import get_dirs

if __name__ == '__main__':
    
    defense_type = 'vib'
    
    dirs = get_dirs(defense_type)
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    model_name = 'vgg16'
    dataset_name = 'celeba'
    batch_size = 64
    epoch_num = 50
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    beta = 0.003
    
    device = 'cuda'
    args = VibTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type=defense_type,
        tqdm_strategy=TqdmStrategy.ITER,
        device=device,
        beta = beta
    )
    
    model = get_model(model_name, dataset_name, device, backbone_pretrain=True, defense_type=defense_type)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=lr, 
                                momentum=momentum, 
                                weight_decay=weight_decay)
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, defense_type)
    
    trainer = VibTrainer(args, folder_manager, model, optimizer, None, beta=beta)
    
    from torchvision.datasets import ImageFolder
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=ToTensor())
    trainloader = DataLoader(trainset, batch_size, shuffle=True, pin_memory=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True)
    
    trainer.train(trainloader, testloader)
