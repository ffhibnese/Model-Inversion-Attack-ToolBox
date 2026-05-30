import sys
import os
import time

sys.path.append("../../../src")

import torch
from torch import nn
from torch.utils.data import DataLoader
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

from modelinversion.models import (
    TorchvisionClassifierModel,
    VibWrapper,
    BiDOWrapper,
    LoraWrapper,
    NeckWrapper,
)
from modelinversion.train import SimpleTrainConfig, SimpleTrainer
from modelinversion.utils import (
    Logger,
    LabelSmoothingCrossEntropyLoss,
    freeze_front_layers,
)
from modelinversion.datasets import CelebA224
import torchvision

torchvision.models.vit_b_16
if __name__ == "__main__":

    num_classes = 1000
    model_name = "vit_b_16"
    lora_dim = 2
    save_name = f"facescrub224_{model_name}_tl0.5_all.pth"
    dataset_path = "/mnt/data/<usrname>/datasets/pre_celeba_high"
    experiment_dir = f"../result_classifier/train_facescrub224_{model_name}_tl_<usrname>"
    backbone_path = "/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/pretrain_vit_b_16/pretrain_vit_b_16.pth"

    batch_size = 128
    epoch_num = 100

    device_ids_str = "4,6"
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r"%Y%m%d_%H%M", time.localtime(time.time()))
    logger = Logger(experiment_dir, f"train_gan_{now_time}.log")

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    model = TorchvisionClassifierModel(
        model_name, num_classes=num_classes, weights="DEFAULT", resolution=224
    )
    state_dict = torch.load(backbone_path, map_location="cpu")["state_dict"]
    # del state_dict['model.head.weight']
    # del state_dict['model.head.bias']
    # del state_dict['model.classifier.bias']
    new_state_dict = {}
    for k, v in state_dict.items():
        if "heads" in k.lower():
            continue
        new_state_dict[k] = v
    load_res = model.load_state_dict(new_state_dict, strict=False)
    print(load_res)
    # model = LoraWrapper(model, lora_dim=lora_dim)
    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    # model = NeckWrapper(model, neck_dim=100)
    freeze_front_layers(model, 0.5)
    model = nn.DataParallel(model, device_ids=gpu_devices)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # prepare dataset

    train_dataset = CelebA224(
        os.path.join(dataset_path, "private_train"),
        output_transform=Compose(
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
    test_dataset = CelebA224(
        os.path.join(dataset_path, "private_test"),
        # train=False,
        output_transform=Compose(
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

    config = SimpleTrainConfig(
        experiment_dir=experiment_dir,
        save_name=save_name,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedular,
        loss_fn="ce",
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, train_loader, test_loader, save_best_ckpts=False)
