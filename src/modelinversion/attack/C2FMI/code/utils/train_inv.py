import torch
from torch import nn, optim
from torch.nn import functional as F
from models.facenet import Facenet
from torchvision.transforms import transforms
from torch.utils import data
from models.predict2feature import predict2feature
import os
from utils.inv_dataset2 import InvDataset


def train(p2f, optimizer, target_model, embed_model, train_loader, test_loader, ep):
    train_loss = 0.
    test_loss  = 0.
    #---------------------------------------------------
    # train
    p2f.train()
    for i,x in enumerate(train_loader):
        x = x.cuda()
        with torch.no_grad():
            before_norm1, _ = target_model.forward_feature(x)
            pre1 = target_model.forward_classifier(before_norm1)
            predict  = F.softmax(pre1, dim=1)
            feature, feature_norm = embed_model.forward_feature(x)
            pre2 = embed_model.forward_classifier(feature)

        map_feat = p2f(predict)
        # mse_loss     = F.mse_loss(map_feat, feature_norm)
        mse_loss     = F.mse_loss(map_feat, pre2)
        train_loss += mse_loss.item()
        avg_train_l = train_loss/(i+1)
        loss = mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch:{ep}, step:{i}, train_loss:{avg_train_l}')

    #---------------------------------------------------
    # test
    p2f.eval()
    for i,x in enumerate(test_loader):
        x = x.cuda()
        with torch.no_grad():
            before_norm1, _ = target_model.forward_feature(x)
            pre1 = target_model.forward_classifier(before_norm1)
            predict  = F.softmax(pre1, dim=1)
            feature, feature_norm = embed_model.forward_feature(x)
            pre2 = embed_model.forward_classifier(feature)

            map_feat = p2f(predict)
            # loss2    = F.mse_loss(map_feat, feature_norm)
            loss2    = F.mse_loss(map_feat, pre2)
            test_loss += loss2.item()
            avg_test_l = test_loss/(i+1)

        print(f'epoch:{ep}, step:{i}, test_loss:{avg_test_l}')
    return avg_test_l


if __name__ == '__main__':
    #---------------------------------------------------
    # parameters
    pre_train_pt  = ''
    device        = 'cuda'
    img_size      = [160,160]
    lr            = 0.001  # 0.001
    batch_size    = 128
    num_classes   = 526
    epoch         = 20

    # training data .txt with each row format:
    # label;abs_dir (0;/home/user1/data/a.jpg)
    train_path    = 'train_path_full.txt'

    # path of the target model
    tar_pth       = 'trained_models/FaceScrub-MobileNet-Train_Acc0.9736-Val_Acc0.9613.pth'

    # path of the embedding model
    emb_pth       = 'trained_models/casia-InceptionResnet-Train_Acc0.984-Val_Acc0.971.pth'

    emb_backbone  = 'inception_resnetv1'
    tar_backbone  = 'mobile_net'

    tar_classes   = NUM_TAR
    emb_classes   = NUM_EMB

    # mask
    trunc         = 1  # full=526

    ck_mid_name   = f'pre2feat_FM2CI_keep{trunc}'
    #---------------------------------------------------

    target_model = Facenet(backbone=tar_backbone, num_classes=tar_classes)
    target_model.load_state_dict(torch.load(tar_pth, map_location='cpu'), strict=True)
    target_model.to(device)
    target_model.eval()

    embed_model = Facenet(backbone=emb_backbone, num_classes=emb_classes)
    embed_model.load_state_dict(torch.load(emb_pth, map_location='cpu'), strict=True)
    embed_model.to(device)
    embed_model.eval()

    model = predict2feature(tar_classes, trunc)
    if pre_train_pt != '':
        model.load_state_dict(torch.load(pre_train_pt, map_location='cpu')['map'], strict=False)
    p2f   = nn.DataParallel(model).to(device)
    optimizer  = optim.Adam(p2f.parameters(), lr=lr, weight_decay=0.000) #  weight_decay=0.00002
    optim_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    #---------------------------------------------------
    # data loader
    with open(train_path, 'r') as f:
        list_path = f.readlines()
    f.close()
    train_data_path = list_path[:150000]
    test_data_path  = list_path[150000:160000]
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )
    train_dataset = InvDataset(train_data_path, transform)
    test_dataset = InvDataset(test_data_path, transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=4)
    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                  drop_last=True)

    #---------------------------------------------------
    # training
    for ep in range(epoch):
        val_loss = train(p2f, optimizer, target_model, embed_model, train_loader, test_loader, ep)
        optim_scheduler.step()
        if (ep+1)%4 == 0:
            save_path = f'checkpoint/{str(ep+1)}_{ck_mid_name}_loss_{round(val_loss,4)}.pt'
            torch.save(
                {
                    'map': p2f.module.state_dict(),
                },
                save_path
            )
