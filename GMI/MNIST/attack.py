import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
num_classes = 10
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)

def inversion(G, D, T, E, iden, lr=1e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)
	
	for random_seed in range(5):
		tf = time.time()
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			out = T(fake)[-1]
			
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)
			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(fake_img)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(fake)[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				z_hat[i, :] = z[i, :]
			if eval_iden[i].item() == gt:
				cnt += 1
			
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

	correct = 0
	for i in range(bs):
		gt = iden[i].item()
		if max_iden[i].item() == gt:
			correct += 1
	
	acc = correct * 1.0 / bs
	print("Acc:{:.2f}".format(acc))

if __name__ == "__main__":
	target_path = "target_model_path"
	
	T = classify.MCNN(num_classes)
	T = nn.DataParallel(T).cuda()
	ckp_T = torch.load(target_path)['state_dict']
	utils.load_my_state_dict(T, ckp_T)

	e_path = "evaluate_model_path"
	E = classify.SCNN(num_classes)
	E = nn.DataParallel(E).cuda()
	ckp_E = torch.load(e_path)['state_dict']
	utils.load_my_state_dict(E, ckp_E)

	g_path = "generator_path"
	G = generator.GeneratorMNIST()
	G = nn.DataParallel(G).cuda()
	ckp_G = torch.load(g_path)['state_dict']
	utils.load_my_state_dict(G, ckp_G)

	d_path = "discriminator_path"
	D = discri.DGWGAN32()
	D = nn.DataParallel(D).cuda()
	ckp_D = torch.load(d_path)['state_dict']
	utils.load_my_state_dict(D, ckp_D)

	iden = torch.zeros(5)
	for i in range(5):
	    iden[i] = i+5

	inversion(G, D, T, E, iden)
	