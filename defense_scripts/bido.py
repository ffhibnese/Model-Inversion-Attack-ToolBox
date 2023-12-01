import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.defense.BiDO import BiDOTrainArgs, BiDOTrainer
from modelinversion.utils import FolderManager
from modelinversion.models import get_model
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from development_config import get_dirs

if __name__ == '__main__':
    
    dirs = get_dirs('bido')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, 'bido')
    
    model_name = 'ir152'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.0001
    device = 'cuda:2'
    bido_loss_type = 'hisc'
    batch_size = 64
    
    train_args = BiDOTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='bido',
        device=device,
        bido_loss_type=bido_loss_type,
        # coef_hidden_input=0.05,
        # coef_hidden_output=0.5
    )
    
    model = get_model(model_name, dataset_name, device=device, backbone_pretrain=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.5)
    
    trainer = BiDOTrainer(train_args, folder_manager, model=model, optimizer=optimizer, scheduler=scheduler)
    
    from torchvision.datasets import ImageFolder
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=ToTensor())
    trainloader = DataLoader(trainset, batch_size, shuffle=True, pin_memory=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True)
    
    trainer.train(trainloader, testloader)
