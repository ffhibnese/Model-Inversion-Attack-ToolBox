import torch, os, time, classify, utils
import numpy as np 
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR
root_path = "./target_models"
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
	
    #scheduler = MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)
        
    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()
		
        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
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

        if (epoch+1) % 10 == 0:
            torch.save({'state_dict':model.state_dict()}, os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc))
        #scheduler.step()

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
			
		print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc))
		

	print("Best Acc:{:.2f}".format(best_ACC))
	return best_model, best_ACC
