import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import copy
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, ToPILImage, Lambda, Resize

from modelinversion.defense.BiDO import BiDOTrainArgs, BiDOTrainer
from modelinversion.utils import FolderManager, RandomIdentitySampler
from modelinversion.models import get_model
from development_config import get_dirs


import numpy as np

class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    
    dirs = get_dirs('bido')
    cache_dir, result_dir, ckpt_dir, dataset_dir, defense_ckpt_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir'], dirs['defense_ckpt_dir']
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir, 'bido')
    
    model_name = 'vgg16'
    dataset_name = 'celeba'
    epoch_num = 50
    lr = 0.0001
    device = 'cuda'
    bido_loss_type = 'hisc'
    batch_size = 64
    
    train_args = BiDOTrainArgs(
        model_name=model_name,
        dataset_name=dataset_name,
        epoch_num=epoch_num,
        defense_type='bido',
        device=device,
        bido_loss_type=bido_loss_type,
        coef_hidden_input=0.05,
        coef_hidden_output=0.5
    )
    
    model = get_model(model_name, dataset_name, device=device, backbone_pretrain=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.5)
    
    trainer = BiDOTrainer(train_args, folder_manager, model=model, optimizer=optimizer, scheduler=scheduler)

    trainset = ImageFolder('./dataset/celeba/split/private/train', transform=Compose([
        RandomHorizontalFlip(p=0.5), ToTensor()
    ]))
    print(trainset[0][0].shape)
    train_sampler = RandomIdentitySampler(trainset, batch_size, 4)
    trainloader = DataLoader(trainset, batch_size, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    testset = ImageFolder('./dataset/celeba/split/private/test', transform=ToTensor())
    testloader = DataLoader(testset, batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    trainer.train(trainloader, testloader)
