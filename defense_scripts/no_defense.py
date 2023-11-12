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

if __name__ == '__main__':
    
    model_name = 'vgg16'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.01
    
    device = 'cuda'
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
        lr=lr
    )
    
    folder_manager = FolderManager('./checkpoints', None, None, './results/no_defense', './checkpoints_defense')
    
    trainer = RegTrainer(args, folder_manager, model, optimizer, None)
    
    from torchvision.datasets import ImageFolder
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=ToTensor())
    trainloader = DataLoader(trainset, 32, shuffle=True, pin_memory=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, 32, shuffle=False, pin_memory=True)
    
    trainer.train(trainloader, testloader)
