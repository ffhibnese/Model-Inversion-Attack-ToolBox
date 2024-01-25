import math
import pickle
import sys
import os
from collections import defaultdict
from typing import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from ...utils import set_random_seed, safe_save, DictAccumulator
from ..base import BaseAttackConfig
from ..base import BaseAttacker, BaseSingleLabelAttacker
from ...foldermanager import FolderManager
from ...models import *

@dataclass
class PPAAttackConfig(BaseAttackConfig):
    
    stylegan_resp_dir: str = '.'
    stylegan_file_path: str = 'ffhq.pkl'
    stylegan_dataset: str = 'ffhq'
    num_candidate: int = 200
    
    init_select_batch_size: int = 25
    init_select_num: int = 5000
    
    # final selection
    samples_per_targets: int = 50
    final_select_iters: int = 100
    
    single_w: bool = True
    
    seed: int = 42
    truncation_psi: float = 0.5
    truncation_cutoff: int = 8
    lr = 0.005
    
    
    num_epochs: int = 200
    
    init_select_transform: Callable = None
    attack_transform: Compose = None
    to_result_transform: Compose = None
    to_eval_transform: Compose = None
    final_select_transform: Compose = None

# TODO; clip images, discri, add clip in transform

class PPAAttacker(BaseSingleLabelAttacker):
    
    def __init__(self, config: PPAAttackConfig) -> None:
        super().__init__(config)
        self.config: PPAAttackConfig
        
    def get_tag(self) -> str:
        return f'{self.config.stylegan_dataset}_{self.config.dataset_name}_{self.config.target_name}'
        
    def prepare_attack(self):
        config: PPAAttackConfig = self.config
        
        set_random_seed(config.seed)
        
        sys.path.insert(0, config.stylegan_resp_dir)
        with open(config.stylegan_file_path, 'rb') as f:
            dic = pickle.load(f)
            G = dic['G_ema']
        
        num_ws = G.num_ws
        synthesis = G.synthesis
        synthesis.num_ws = num_ws
        
        mapping_module = G.mapping.to(config.device)
        self.synthesis_module = synthesis.to(config.device)
        # mapping_module.eval()
        # self.synthesis_module.eval()
        
        self.num_ws = num_ws
        self.z_dim = G.z_dim
        
        print('create initial w')
        
        self._create_initial_ws(self.target_labels, mapping_module)
        del mapping_module

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis_module(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis_module(w, noise_mode='const', force_fp32=True)
        return imgs
        
    def _get_init_w_path(self, target):
        config = self.config
        w_tag = 'init_single_w' if config.single_w else 'init_multi_w'
        w_dir = os.path.join(self.folder_manager.config.cache_dir, w_tag, f'{config.num_candidate}')
        os.makedirs(w_dir, exist_ok=True)
        return os.path.join(w_dir, f'w_{target}.npy')
    
    @torch.no_grad()
    def _create_initial_ws(self, targets, mapping_module):
        config: PPAAttackConfig = self.config
        
        gen_targets = [target for target in targets if not os.path.exists(self._get_init_w_path(target))]
        
        if len(gen_targets) == 0:
            return
        
        print('generate init w')
        # z = torch.randn(config.init_select_num, self.z_dim)
        z = torch.from_numpy(
            np.random.RandomState(config.seed).randn(config.init_select_num, self.z_dim)
        )
        c = None
        
        candidates = []
        confidences = []
        
        for start_idx in tqdm(range(0, config.init_select_num, config.init_select_batch_size)):
            end_idx = min(start_idx + config.init_select_batch_size, config.init_select_num)
            z_batch = z[start_idx: end_idx].to(config.device)
            w_batch = mapping_module(z_batch, c, truncation_psi=config.truncation_psi, truncation_cutoff=config.truncation_cutoff)
            img_batch = self.synthesize(w_batch, self.num_ws)
            if config.init_select_transform:
                aug_imgs_batch = config.init_select_transform(img_batch)
            else:
                aug_imgs_batch = [img_batch]
            
            target_score = 0
            for imgs in aug_imgs_batch:
                target_score += self.T(imgs).result.softmax(dim=-1)
            target_score /= len(aug_imgs_batch)
            
            candidates.append(w_batch.cpu())
            confidences.append(target_score.cpu())
        
        candidates = torch.cat(candidates, dim=0)
        confidences = torch.cat(confidences, dim=0)
        for target in gen_targets:
            confs, indice = torch.topk(confidences[:, target], k=config.num_candidate)
            target_candidate = candidates[indice]
            if config.single_w:
                target_candidate = target_candidate[:,0].unsqueeze(1)
            # torch.save()
            save_path = self._get_init_w_path(target)
            print(f'save init w for target {target}')
            np.save(save_path, target_candidate.numpy())
    
    def _to_onehot(self, y, num_classes):
        """ 1-hot encodes a tensor """
        # return torch.squeeze(torch.eye(num_classes)[y.cpu()], dim=1)
        return torch.zeros((len(y), num_classes)).to(y.device).scatter_(1, y.reshape(-1, 1), 1.)
    
    def _get_loss(self, pred, labels, eps=1e-4):
        u = pred / torch.norm(pred, p=1, dim=-1, keepdim=True)
        v = self._to_onehot(labels, pred.shape[-1])
        v = torch.clip(v, 0, 1-eps)
        
        u_norm2 = torch.norm(u, p=2, dim=1) ** 2
        v_norm2 = torch.norm(v, p=2, dim=1) ** 2
        diff_norm2 = torch.norm(u - v, p=2, dim=1) ** 2
        
        delta = 2 * diff_norm2 / ((1-u_norm2) * (1-v_norm2))
        loss = torch.arccosh(1 + delta)
        return loss.mean()
        
    
    def _optimize(self, w: torch.Tensor, target: int):
        config = self.config
        w.requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=config.lr, betas=[0.1, 0.1])
        
        # TODO: scheduler
        
        config: PPAAttackConfig = self.config
        
        for epoch in tqdm(range(config.num_epochs)):
            imgs = self.synthesize(w, num_ws = self.num_ws)
            
            if config.attack_transform is not None:
                imgs = config.attack_transform(imgs)
                
            outputs = self.T(imgs).result
            labels = torch.ones((len(outputs),), dtype=torch.long, device=config.device) * target
            loss = self._get_loss(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return w.detach()
        
    # @torch.no_grad()
    # def _generate_result_images(self, w: torch.Tensor):
    #     if w.shape[1] == 1:
    #         w = torch.repeat_interleave(w, repeats=self.num_ws, dim=1)
            
    #     imgs = []
        
    #     for i in range(math.ceil(w.shape[0] / self.batch_size)):
    #         w_batch = w[i * self.batch_size:min(len(w), (i + 1) * self.batch_size)]
    #         w_batch = w_batch.to(self.config.device)
    #         imgs_batch = self.synthesize(w_batch, self.num_ws)
        
    #         if self.config.to_result_transform is not None:
    #             imgs_batch = self.config.to_result_transform(imgs_batch)
                
    #         imgs.append(imgs_batch.cpu())
            
    #     return imgs
        
    def _calc_acc(self, imgs, label, model):
        imgs = imgs.to(self.config.device)
        pred_res = model(imgs).result
        _, top5idx = torch.topk(pred_res, k=5, dim=-1)
        eq = (top5idx == label).float()
        
        acc5 = eq.sum() / len(imgs)
        acc = eq[:, 0].sum() / len(imgs)
        return acc.item(), acc5.item()
            
        
    def attack_step(self, target):
        
        config = self.config
        
        w_init_path = self._get_init_w_path(target)
        w = np.load(w_init_path)
        
        w_optimized = []
        
        print(f'start optimize')
        for start_idx in range(0, w.shape[0], self.batch_size):
            end_idx = min(w.shape[0], start_idx + self.batch_size)
            w_batch = torch.from_numpy(w[start_idx: end_idx]).to(config.device)
            w_batch_optimized = self._optimize(w_batch, target).cpu()
            torch.cuda.empty_cache()
            w_optimized.append(w_batch_optimized)
        
        # final selection
        
        # eval_accs = []
        # eval_acc5s = []
        
        print(f'final selection')
        
        with torch.no_grad():
            scores = []
            T_imgs = []
            E_imgs = []
            
            unfilter_accumulator = DictAccumulator()
            
            for w_batch_optimized in w_optimized:
                w_batch_optimized = w_batch_optimized.to(config.device)
                imgs_batch = self.synthesize(w_batch_optimized, self.num_ws)
                T_imgs_batch = imgs_batch
                E_imgs_batch = imgs_batch
                if self.config.to_result_transform is not None:
                    T_imgs_batch = self.config.to_result_transform(imgs_batch)
                if self.config.to_eval_transform is not None:
                    E_imgs_batch = self.config.to_eval_transform(imgs_batch)
                    
                T_acc_batch, T_acc5_batch = self._calc_acc(T_imgs_batch, target, self.T)
                E_acc_batch, E_acc5_batch = self._calc_acc(E_imgs_batch, target, self.E)
                unfilter_accumulator.add({
                    'target acc': T_acc_batch,
                    'target acc5': T_acc5_batch,
                    'eval acc': E_acc_batch,
                    'eval acc5': E_acc5_batch
                }, add_num=len(imgs_batch))
                
                
                T_imgs.append(T_imgs_batch.cpu())
                E_imgs.append(E_imgs_batch.cpu())
                scores_batch = 0
                for i in tqdm(range(config.final_select_iters)):
                    # scores += self._generate_label_scores(imgs_batch, target, self.T, config.final_select_transform)
                    aug_imgs_batch = imgs_batch if config.final_select_transform is None else config.final_select_transform(imgs_batch)
                    scores_batch += torch.softmax(self.T(aug_imgs_batch).result, dim=-1)[..., target].detach().cpu()
                scores_batch /= config.final_select_iters
                scores.append(scores_batch)
                
            w_optimized = torch.cat(w_optimized, dim=0)
            T_imgs = torch.cat(T_imgs, dim=0)
            E_imgs = torch.cat(E_imgs, dim=0)
            scores = torch.cat(scores, dim=0)
            
            top_scores, top_indice = torch.topk(scores, k=config.samples_per_targets, dim=-1)
            top_ws = w_optimized[top_indice]
            top_T_imgs = T_imgs[top_indice]
            top_E_imgs = E_imgs[top_indice]
            
            safe_save(top_ws, os.path.join(self.folder_manager.config.result_dir, 'final_w'), f'{target}.pt')
            self.folder_manager.save_result_images(top_E_imgs, target)
            
            T_acc, T_acc5 = self._calc_acc(top_T_imgs, target, self.T)
            E_acc, E_acc5 = self._calc_acc(top_E_imgs, target, self.E)
                
            print(f'>> label {target}')
            print('unfiltered')
            for k, v in unfilter_accumulator.avg().items():
                print(f'{k}: {v}')
            print('Filtered:')
            print(f'target acc: {T_acc} \ntarget acc5: {T_acc5}')
            print(f'eval acc: {E_acc} \neval acc5: {E_acc5}')
                
        return {
            'acc': E_acc,
            'acc5': E_acc5
        }