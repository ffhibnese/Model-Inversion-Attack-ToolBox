import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

from modelinversion.defense import *
from modelinversion.utils import FolderManager, RandomIdentitySampler
from modelinversion.models import get_model
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, ToPILImage, Lambda, Resize

if __name__ == '__main__':
    
    model_name = 'vit'
    dataset_name = 'celeba'
    epoch_num = 500
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
    
    # model = get_model(model_name, dataset_name, device, backbone_pretrain=True)
    from modelinversion.models.vit.vit import ViT
    model = ViT(
        image_size=64, 
        patch_size=8,
        num_classes=1000,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr
    # )e
    # state_dict = torch.load(f'checkpoints/target_eval/celeba/vit_celeba.pt', map_location=device)['state_dict']
    # model.load_state_dict(state_dict)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    folder_manager = FolderManager('./checkpoints', None, None, './results/no_defense', './checkpoints_defense')
    
    trainer = RegTrainer(args, folder_manager, model, optimizer, None)
    
    from torchvision.datasets import ImageFolder
    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=Compose([
        RandomHorizontalFlip(p=0.5), ToTensor()
    ]))
    # print(trainset[0][0].shape)
    
    batch_size = 64
    train_sampler = RandomIdentitySampler(trainset, batch_size, 4)
    trainloader = DataLoader(trainset, batch_size, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    trainer.train(trainloader, testloader)
    # test_acc, = trainer._test_loop(testloader)
    # print(test_acc)