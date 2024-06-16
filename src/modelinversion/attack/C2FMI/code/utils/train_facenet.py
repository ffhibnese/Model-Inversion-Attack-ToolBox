import torch
from torch import nn
from models.facenet import Facenet
from utils.weights_init import weights_init
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.triplet_loss import triplet_loss
from utils.dataloder_face import FacenetDataset, collate_fc
from utils.update_one_epoch_facenet import one_epoch_update
import argparse


def get_num_classes(annotation_path):
    # return total classes of training data
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


if __name__ == "__main__":

    # ---------------------------------------------------
    # parameters
    pretrained_dir = None
    input_shape = [160, 160, 3]
    backbone = ''  # 'mobile_net' or 'inception_resnetv1'
    saved_net_path = ''  # better give a corresponding pre-trained model's path
    annotation_path = ''  # run annotation_face_train.py
    annotation_path_val = ''  # run annotation_face_train.py
    # ---------------------------------------------------

    freeze_backbone = True  # freeze backbone in training stage 1
    num_workers = 4
    num_classes = get_num_classes(annotation_path)
    dataset_collate = collate_fc()

    net = Facenet(
        backbone=backbone, num_classes=num_classes, pretrained_dir=pretrained_dir
    )
    if pretrained_dir is None:
        weights_init(net)
    if saved_net_path != '':
        print(f'Load Weights from {saved_net_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(saved_net_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)

    using_gpu = False
    if torch.cuda.is_available():
        using_gpu = True
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True
    if not using_gpu:
        print('---------Not using GPU----------')
    else:
        print('---------Using GPU---------')

    trip_loss = triplet_loss()

    # If using CAISA_WebFace dataset
    # 5% 验证，95% 训练
    # val_split = 0.05
    # with open(annotation_path,"r") as f:
    #     lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val
    #
    # lines_train = lines[:num_train]
    # lines_val   = lines[num_train:]

    # If using FaceScrub
    with open(annotation_path, "r") as f:
        lines_train = f.readlines()
    f.close()
    with open(annotation_path_val, "r") as f:
        lines_val = f.readlines()
    f.close()
    num_train = len(lines_train)
    num_val = len(lines_val)

    # stage 1：freeze backbone
    if True:
        lr = 1e-3
        batch_size = 64
        init_epoch = 0
        interval_epoch = 10

        epoch_step = num_train // batch_size  # 一个训练epoch需要的iteration次数
        epoch_step_val = num_val // batch_size  # 一个验证epoch需要的iteration次数

        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = FacenetDataset(input_shape, lines_train, num_train, num_classes)
        val_dataset = FacenetDataset(input_shape, lines_val, num_val, num_classes)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_collate,
        )

        if freeze_backbone:
            if not using_gpu:
                for param in net.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in net.module.backbone.parameters():
                    param.requires_grad = False

        for epoch in range(init_epoch, interval_epoch):
            one_epoch_update(
                net,
                trip_loss,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                train_loader,
                val_loader,
                interval_epoch,
                batch_size,
                using_gpu,
            )
            lr_scheduler.step()

    # stage 2：unfreeze backbone (change False to True)
    if False:
        lr = 1e-4
        batch_size = 32
        final_epoch = 10

        epoch_step = num_train // batch_size  # 一个训练epoch需要的iteration次数
        epoch_step_val = num_val // batch_size  # 一个验证epoch需要的iteration次数

        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = FacenetDataset(input_shape, lines_train, num_train, num_classes)
        val_dataset = FacenetDataset(input_shape, lines_val, num_val, num_classes)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_collate,
        )

        if not using_gpu:
            for param in net.backbone.parameters():
                param.requires_grad = True
        else:
            for param in net.module.backbone.parameters():
                param.requires_grad = True

        for epoch in range(interval_epoch, final_epoch):
            one_epoch_update(
                net,
                trip_loss,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                train_loader,
                val_loader,
                final_epoch,
                batch_size,
                using_gpu,
            )
            lr_scheduler.step()
