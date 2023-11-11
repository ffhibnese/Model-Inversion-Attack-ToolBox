import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.defense.base import *
from modelinversion.utils import DefenseFolderManager
from modelinversion.models import get_model
from torchvision.transforms import ToTensor
import torch

if __name__ == '__main__':
    
    model_name = 'vgg16'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.01
    
    device = 'cuda'
    args = TrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='no_defense',
        device=device
    )
    
    model = get_model(model_name, dataset_name, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )
    
    folder_manager = DefenseFolderManager('./checkpoints', None, None, './result/no_defense', './checkpoints_defense')
    
    trainer = RegTrainer(args, folder_manager, model, optimizer, None, nn.CrossEntropyLoss().to(device))
    
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder('./dataset/celeba/split/private/train', transform=ToTensor())
    dataloader = DataLoader(dataset, 32, shuffle=True, pin_memory=True)
    
    trainer.train(dataloader)
