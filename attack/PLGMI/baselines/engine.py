import numpy as np
import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR

import classify
import utils

root_path = "./target_model_at_eps4"
model_path = os.path.join(root_path, "target_ckp")

device = "cuda"


def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        out_prob = model(img)[-1]
        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def train_reg(args, model, criterion, optimizer, trainloader, testloader, n_epochs):
    best_ACC = 0.0
    model_name = args['dataset']['model_name']

    # scheduler = MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)  # 交叉熵
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)

        if (epoch + 1) % 10 == 0:
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc, test_acc))
        # scheduler.step()

    print("Best Acc:{:.2f}".format(best_ACC))
    return best_model, best_ACC


def train_vib(args, model, criterion, optimizer, trainloader, testloader, n_epochs):
    best_ACC = 0.0
    model_name = args['dataset']['model_name']

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0

        for i, (img, iden) in enumerate(trainloader):
            img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            ___, out_prob, mu, std = model(img, "train")
            cross_loss = criterion(out_prob, one_hot)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_loss, test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc, test_acc))

    print("Best Acc:{:.2f}".format(best_ACC))
    return best_model, best_ACC


import pgd


def adjust_learning_rate(args, optimizer, epoch):
    """decrease the learning rate"""
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    if epoch >= 100:
        lr = args[model_name]['lr'] * 0.1
    if epoch >= 150:
        lr = args[model_name]['lr'] * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def robust_test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    # attack = pgd.PGD(model, eps=8/255, alpha=2./255, steps=10, random_start=True)
    attack = pgd.PGD(model, eps=4 / 255, alpha=4. / 255 / 7, steps=7, random_start=True)

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        adv_img = attack(img, iden)

        out_prob = model(adv_img)[-1]
        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def train_reg_at(args, model, criterion, optimizer, trainloader, testloader, n_epochs):
    best_ACC = 0.0
    model_name = args['dataset']['model_name']

    # scheduler = MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)

    attack = pgd.PGD(model, eps=4 / 255, alpha=4. / 255 / 7, steps=7, random_start=True)
    # attack = pgd.PGD(model, eps=8/255, alpha=2./255, steps=10, random_start=True)

    for epoch in range(n_epochs):
        adjust_learning_rate(args, optimizer, epoch)
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        # model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)

            bs = img.size(0)

            # print(img.shape)
            # print(iden.shape)

            iden = iden.view(-1)

            model.eval()
            adv_img = attack(img, iden)
            # adv_img = img

            model.train()
            # print(iden.shape)

            feats, out_prob = model(adv_img)
            cross_loss = criterion(out_prob, iden)  # 交叉熵
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs
            # print(ACC/cnt)

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader)
        robust_test_acc = robust_test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)

        # if (epoch+1) % 10 == 0:
        torch.save({'state_dict': model.state_dict()}, os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))

        print(
            "Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}\tRobust Test Acc:{:.2f}".format(
                epoch, interval, train_loss, train_acc, test_acc, robust_test_acc))
        # scheduler.step()

    print("Best Acc:{:.2f}".format(best_ACC))
    return best_model, best_ACC
