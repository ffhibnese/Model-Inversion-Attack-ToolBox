import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .config import KEDMIAttackConfig
from ..base import BaseAttacker
from .code.generator import Generator
from .code.discri import MinibatchDiscriminator

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu
    
def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))

class KEDMIAttacker(BaseAttacker):
    
    def __init__(self, config: KEDMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}_{config.gan_target_name}'
        super().__init__(config)
        self.config: KEDMIAttackConfig
        
    def get_tag(self) -> str:
        return self._tag
    
    def prepare_attack(self):
        config = self.config
        
        self.G = Generator(config.z_dim).to(config.device)
        self.D = MinibatchDiscriminator().to(config.device)
        
        self.folder_manager.load_state_dict(self.G, 
                                   ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_G.tar'],
                                   device=config.device)
        self.folder_manager.load_state_dict(self.D, 
                                    ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_D.tar'],
                                    device=config.device)
        
        self.G.eval()
        self.D.eval()
        
    
        
    def attack_step(self, iden) -> dict:
        
        config = self.config
        device = config.device
        folder_manager = self.folder_manager
        
        
        iden = iden.view(-1).long().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        bs = iden.shape[0]

        # NOTE
        mu = Variable(torch.zeros(bs, 100), requires_grad=True)
        log_var = Variable(torch.ones(bs, 100), requires_grad=True)

        params = [mu, log_var]
        solver = optim.Adam(params, lr=config.lr)
        
        tf = time.time()
        
        for i in range(config.iter_times):
            z = reparameterize(mu, log_var).to(device)
            fake = self.G(z)
            
            _, label = self.D(fake)

            out = self.T(fake).result
            
            # for p in params:
            #     if p.grad is not None:
            #         p.grad.data.zero_()
            solver.zero_grad()
            
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
            Iden_Loss = criterion(out, iden)

            Total_Loss = Prior_Loss + config.lamda * Iden_Loss

            Total_Loss.backward()
            solver.step()
            
            z = torch.clamp(z.detach(), -config.clip_range, config.clip_range).float()

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if (i + 1) % 300 == 0:
                with torch.no_grad():
                    fake_img = self.G(z.detach())
                    eval_prob = self.E(fake_img).result
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print(
                        "Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, Prior_Loss_val,
                                                                                                    Iden_Loss_val, acc))
        interval = time.time() - tf
        print("Time:{:.2f}".format(interval))
        
        with torch.no_grad():
            res = []
            res5 = []
            seed_acc = torch.zeros((bs, 5))
            for random_seed in range(config.gen_num_per_target):
                tf = time.time()
                z = reparameterize(mu, log_var).to(device)
                fake = self.G(z)
                # score = T(fake).result
                eval_prob = self.E(fake).result
                eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

                cnt, cnt5 = 0, 0
                for i in range(bs):
                    gt = iden[i].item()
                    sample = fake[i]
                    
                    folder_manager.save_result_image(sample, gt)

                    if eval_iden[i].item() == gt:
                        seed_acc[i, random_seed] = 1
                        cnt += 1
                        best_img = self.G(z)[i]
                        folder_manager.save_result_image(best_img, gt, folder_name='success_imgs')
                    
                    _, top5_idx = torch.topk(eval_prob[i], 5)
                    if gt in top5_idx:
                        cnt5 += 1

                interval = time.time() - tf
                print("Time:{:.2f}\tSeed:{}\tAcc:{:.2f}\t".format(interval, random_seed, cnt * 1.0 / bs))
                res.append(cnt * 1.0 / bs)
                res5.append(cnt5 * 1.0 / bs)

                torch.cuda.empty_cache()

        acc, acc_5 = statistics.mean(res), statistics.mean(res5)
        acc_var = statistics.variance(res)
        acc_var5 = statistics.variance(res5)
        print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

        return {
            'acc': acc,
            'acc5': acc_5,
            'acc_var': acc_var,
            'acc5_var': acc_var5
        }