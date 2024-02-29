

import statistics
import time
from collections import OrderedDict
from tqdm import tqdm

import torch
from kornia import augmentation

from ..base import BaseAttacker
from .code.reconstruct import inversion
from .config import PLGMIAttackConfig
from ...foldermanager import FolderManager
from .code.models.generators.resnet64 import ResNetGenerator
from ...models import *
from ...utils import *


class PLGMIAttacker(BaseAttacker):
    
    def __init__(self, config: PLGMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}_{config.gan_target_name}'
        super().__init__(config)
        self.config: PLGMIAttackConfig
        self.loss_fn = ClassifyLoss(config.inv_loss_type)
        
    def get_tag(self) -> str:
        return self._tag
        
    def prepare_attack(self):
        config: PLGMIAttackConfig = self.config
        self.G = ResNetGenerator(num_classes=NUM_CLASSES[config.dataset_name], distribution=config.gen_distribution).to(self.config.device)
        if 'ffhq' in config.gan_dataset_name:
            path = '/data/yuhongyao/papar_codes/PLG-MI-Attack/PLG_high_test/gen_14_iter_0014000.pth.tar'
            
        else:
            path = '/data/yuhongyao/papar_codes/PLG-MI-Attack/PLG_high_metfaces_test/gen_29_iter_0029000.pth.tar'
        print(f'load from {path}')
        self.G.load_state_dict(torch.load(path)['model'])
        # self.folder_manager.load_state_dict(
        #     self.G, 
        #     ['PLGMI_high', f'plgmi_high_{config.gan_dataset_name}_{config.gan_target_name}_{config.dataset_name}_G.pt'], 
        #     device=config.device
        # )
        # self.G.load_state_dict()
        self.G = nn.DataParallel(self.G)
        
    def get_loss(self, fake, iden):
        aug_list = self.config.attack_transform
        out1 = self.T(aug_list(fake)).result
        out2 = self.T(aug_list(fake)).result

        inv_loss = self.loss_fn(out1, iden) + self.loss_fn(out2, iden)
        return inv_loss
        
        
    def attack_step(self, iden):
        
        config = self.config
        device = config.device
        
        bs = iden.shape[0]
        iden = iden.view(-1).long().to(config.device)

        flag = torch.zeros(bs)
        no = torch.zeros(bs)  # index for saving all success attack images

        res = []
        res5 = []
        seed_acc = torch.zeros((bs, config.gen_num_per_target))

        

        for random_seed in range(config.gen_num_per_target):
            tf = time.time()
            r_idx = random_seed

            set_random_seed(random_seed)

            # z = utils.sample_z(
            #     bs, config.gen_dim_z, config.device, config.gen_distribution
            # )
            z = torch.randn((bs, config.gen_dim_z), device=device, requires_grad=True)
            # z.requires_grad = True

            optimizer = torch.optim.Adam([z], lr=config.lr)

            for i in tqdm(range(config.iter_times)):
                fake = self.G(z, iden)
                inv_loss = self.get_loss(fake, iden)

                optimizer.zero_grad()
                inv_loss.backward()
                optimizer.step()

                inv_loss_val = inv_loss.item()

                if (i + 1) % 100 == 0:
                    with torch.no_grad():
                        fake_img = self.G(z, iden)
                        eval_prob = self.E(fake_img).result
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                        print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

            with torch.no_grad():
                fake = self.G(z, iden)
                eval_prob = self.E(fake).result
                eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

                cnt, cnt5 = 0, 0
                for i in range(bs):
                    gt = iden[i].item()
                    sample = fake[i]
                    self.folder_manager.save_result_image(sample, gt, save_tensor=True)

                    if eval_iden[i].item() == gt:
                        seed_acc[i, r_idx] = 1
                        cnt += 1
                        flag[i] = 1
                        best_img = fake[i]
                        self.folder_manager.save_result_image(best_img, gt, folder_name='success_imgs')
                        
                    _, top5_idx = torch.topk(eval_prob[i], 5)
                    if gt in top5_idx:
                        cnt5 += 1

                interval = time.time() - tf
                print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
                res.append(cnt * 1.0 / bs)
                res5.append(cnt5 * 1.0 / bs)
                if config.device == 'cuda':
                    torch.cuda.empty_cache()

        acc, acc_5 = statistics.mean(res), statistics.mean(res5)
        acc_var = statistics.variance(res)
        acc_var5 = statistics.variance(res5)
        print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

        # return acc, acc_5, acc_var, acc_var5
        # return {
        #     'acc': acc,
        #     'acc5': acc_5,
        #     'acc_var': acc_var,
        #     'acc5_var': acc_var5
        # }
        return OrderedDict(
            acc = acc,
            acc5 = acc_5,
            acc_var = acc_var,
            acc5_var = acc_var5
        )
        