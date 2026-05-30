import sys
import os
import time
import random

sys.path.append('../../../src')

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    Resize,
)

from modelinversion.models import TorchvisionClassifierModel, VibWrapper, BiDOWrapper, LoraWrapper
from modelinversion.train import MixTrainConfig, MixTrainer
from modelinversion.utils import Logger, LabelSmoothingCrossEntropyLoss, freeze_front_layers , InverseFocalLoss
from modelinversion.datasets import FaceScrub

class MixDatasetWrapper(Dataset):

    def __init__(self, dataset: ImageFolder, mix_length_ratio = 1, fix_mix=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.mix_length = int(len(dataset) * mix_length_ratio)
        self.fix_mix = fix_mix

        class_indices = [[] for _ in range(len(dataset.classes))]
        self.class_indices = class_indices

        for index, label in enumerate(dataset.targets):
            class_indices[label].append(index)

        if self.fix_mix:
            # choices = list(range(len(dataset)))
            self.mix_indices_1 = random.choices(dataset.targets, k=self.mix_length)

            self.mix_indices_2 = [random.choice(class_indices[dataset.targets[idx]]) for idx in self.mix_indices_1]

    def __getitem__(self, index):
        if index < len(self.dataset):
            img, label = self.dataset[index]
            label = torch.LongTensor([label, 0])
            return img, label
        
        index = index - len(self.dataset)
        if self.fix_mix:
            img1, label1 = self.dataset[self.mix_indices_1[index]]
            img2, label2 = self.dataset[self.mix_indices_2[index]]
        else:
            img1, label1 = self.dataset[random.randint(0, len(self.dataset) - 1)]
            index2 = random.choice(self.class_indices[label1])
            img2, label2 = self.dataset[index2]
            # img2, label2 = self.dataset[random.randint(0, len(self.dataset) - 1)]
        assert label1 == label2, f'dataset mix {label1} != {label2}'
        return (img1 + img2) / 2, torch.LongTensor([label1, 1])


    def __len__(self):
        return len(self.dataset) + self.mix_length
    


def main(fix_mix, mix_ratio):
    num_classes = 530
    model_name = 'resnet152'
    tl_coef = 0.5
    # fix_mix=True
    # mix_ratio = 1
    if fix_mix:
        fix_mix_tag = 'fixmix'
    else:
        fix_mix_tag = 'freemix'
    focal_gamma = 3

    cenls = -0.05
    save_name = f'facescrub224_{model_name}_tl{tl_coef}_focal{focal_gamma}_{fix_mix_tag}{mix_ratio}_cenls{cenls}.pth'
    dataset_path = '/data/<usrname>/datasets/facescrub/'
    experiment_dir = f'../result_classifier/train_facescrub64_{model_name}_tl{tl_coef}_focal{focal_gamma}_{fix_mix_tag}{mix_ratio}_cenls{cenls}'
    backbone_path = '/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/pretrain_resnet152/pretrain_facescrub224_resnet152.pth'

    batch_size = 96
    epoch_num = 100

    device_ids_str = '2'
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = TorchvisionClassifierModel(
        model_name, num_classes=num_classes, weights='DEFAULT', register_last_feature_hook=True
    )
    state_dict = torch.load(backbone_path, map_location='cpu')["state_dict"]
    del state_dict['model.fc.weight']
    del state_dict['model.fc.bias']
    load_res = model.load_state_dict(state_dict, strict=False)
    print(load_res)
    # model = LoraWrapper(model, lora_dim=lora_dim)
    freeze_front_layers(model, ratio=tl_coef)
    # freeze_front_layers(model, ratio=1)
    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # prepare dataset

    train_dataset = FaceScrub(
        dataset_path,
        train=True,
        crop_center=False,
        preprocess_resolution=224,
        transform=Compose(
            [
                ToTensor(),
                RandomResizedCrop(
                    size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
                ),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    train_dataset = MixDatasetWrapper(train_dataset, mix_length_ratio=mix_ratio, fix_mix=fix_mix)
    test_dataset = FaceScrub(
        dataset_path,
        train=False,
        crop_center=False,
        preprocess_resolution=224,
        transform=Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
    )

    # prepare train config

    config = MixTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        mix_loss_fn=InverseFocalLoss(gamma=focal_gamma),
        origin_loss_fn=LabelSmoothingCrossEntropyLoss(cenls)
    )

    trainer = MixTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader)
    # trainer._test_loop(test_loader)

if __name__ == '__main__':
    for mix_ratio in [0.5]:
        main(False, mix_ratio)