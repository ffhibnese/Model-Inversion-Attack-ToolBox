
from ..base import BaseAttacker
from ...foldermanager import FolderManager
from ...models import *
import math
import pickle
import sys
from collections import defaultdict
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass, field
from ..base import BaseAttackConfig
@dataclass
class PPAAttackConfig(BaseAttackConfig):
    
    stylegan_resp_dir: str = '.'
    stylegan_file_path: str = 'ffhq.pkl'
    num_candidate: int = 20
    
    # final selection
    samples_per_targets: int = 10
    final_selection_iters: int = 10
    
    single_w: bool = True
    
    seed: int = 42
    truncation_psi: float = 0.5
    truncation_cutoff: int = 8
    search_space_size: int = 50
    
    num_epochs: int = 20
    
    init_select_transform: Callable = None
    attack_transform: Compose = None
    to_result_transform: Compose = None
    final_select_transform: Compose = None

# TODO; clip images, discri, add clip in transform

class PPAAttacker(BaseAttacker):
    
    def __init__(self, config: PPAAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}'
        super().__init__(config)
        self.config: PPAAttackConfig
        
    def get_tag(self) -> str:
        return self._tag
        
    def prepare_attack(self):
        config: PPAAttackConfig = self.config
        
        sys.path.insert(0, config.stylegan_resp_dir)
        with open(config.stylegan_file_path, 'rb') as f:
            dic = pickle.load(f)
            G = dic['G_ema'] #, dic['D']
            
        G = G.to(config.device)
        G.eval()
        
        num_ws = G.num_ws
        synthesis = G.synthesis
        synthesis.num_ws = num_ws
        
        self.G = G
        self.synthesis = synthesis
        
        self.num_ws = num_ws
        
    # -> (num targets, num candidate)
    @torch.no_grad()
    def _create_initial_ws(self, targets):
        config: PPAAttackConfig = self.config
        
        z = torch.from_numpy(np.random.RandomState(config.seed).randn(config.search_space_size, self.G.z_dim)).to(config.device)
        c = None
        
        confidences = []
        final_candidates = []
        final_confidences = []
        
        candidates = self.G.mapping(z,
                                       c,
                                       truncation_psi=config.truncation_psi,
                                       truncation_cutoff=config.truncation_cutoff).cpu()
        z = z.cpu()
        candidate_dataset = TensorDataset(candidates)
        for w, in tqdm(DataLoader(candidate_dataset, batch_size=self.batch_size, shuffle=False)):
            
            # -> cuda
            w = w.to(config.device)
            imgs = self.synthesis(w, noise_mode = 'const', force_fp32=True)
            if config.init_select_transform is not None:
                aug_imgs = config.init_select_transform(imgs)
            else:
                aug_imgs = [imgs]

            target_conf = 0
            for ims in aug_imgs:
                target_conf += self.T(ims).result.softmax(dim=-1) / len(aug_imgs)
                
            # -> cpu
            confidences.append(target_conf.cpu())
        
        confidences = torch.cat(confidences, dim=0)
        for target in targets:
            confs, indice = torch.topk(confidences[:, target], k=config.num_candidate)
            final_candidates.append(candidates[indice])

        return torch.cat(final_candidates, dim=0)
    
    def _to_onehot(self, y, num_classes):
        """ 1-hot encodes a tensor """
        # return torch.squeeze(torch.eye(num_classes)[y.cpu()], dim=1)
        return torch.zeros((len(y), num_classes)).to(self.config.device).scatter_(1, y.reshape(-1, 1), 1.)
    
    def _get_loss(self, pred, labels, eps=1e-4):
        u = pred / torch.norm(pred, p=1, dim=-1, keepdim=True)
        v = self._to_onehot(labels, pred.shape[-1]) - eps
        
        u_norm2 = torch.norm(u, p=2, dim=-1) ** 2
        v_norm2 = torch.norm(v, p=2, dim=1) ** 2
        diff_norm2 = torch.norm(u - v, p=2, dim=-1) ** 2
        
        delta = 2 * diff_norm2 / ((1-u_norm2) * (1-v_norm2))
        loss = torch.arccosh(1 + delta)
        return loss.mean()
        
    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs
    
    def _optimize(self, w: torch.Tensor, iden):
        w.requires_grad_(True)
        optimizer = torch.optim.Adam([w], betas=[0.1, 0.1])
        
        # TODO: scheduler
        
        config: PPAAttackConfig = self.config
        
        for epoch in tqdm(range(config.num_epochs)):
            imgs = self.synthesize(w, num_ws = self.num_ws)
            
            if config.attack_transform is not None:
                imgs = config.attack_transform(imgs)
                
            outputs = self.T(imgs).result
            loss = self._get_loss(outputs, iden)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return w.detach()
        
    @torch.no_grad()
    def _generate_result_image(self, w: torch.Tensor):
        if w.shape[1] == 1:
            w = torch.repeat_interleave(w, repeats=self.num_ws, dim=1)
            
        imgs = []
        
        for i in range(math.ceil(w.shape[0] / self.batch_size)):
            w_batch = w[i * self.batch_size:min(len(w), (i + 1) * self.batch_size)]
            w_batch = w_batch.to(self.config.device)
            imgs_batch = self.synthesis(w_batch, noise_mode='const', force_fp32=True)
        
            if self.config.to_result_transform is not None:
                imgs_batch = self.config.to_result_transform(imgs_batch)
                
            imgs.append(imgs_batch.cpu())
            
        return torch.cat(imgs, dim=0)
    
    @torch.no_grad()
    def _generate_label_scores(self, imgs: torch.Tensor, label, model, transform=None):
            
        scores = []
        
        for i in range(math.ceil(imgs.shape[0] / self.batch_size)):
            imgs_batch = imgs[i * self.batch_size:min(len(imgs), (i + 1) * self.batch_size)]
            imgs_batch = imgs_batch.to(self.config.device)
            if transform is not None:
                imgs_batch = transform(imgs_batch)
                
            scores_batch = model(imgs_batch).result[..., label]
                
            scores.append(scores_batch.cpu())
            
        return torch.cat(scores, dim=0)
        
    def _calc_acc(self, imgs, label, model):
        pred_res = model(imgs).result
        _, top5idx = torch.topk(pred_res, k=5)
        eq = (top5idx == label).float()
        
        acc5 = eq.sum() / len(imgs)
        acc = eq[:, 0].sum() / len(imgs)
        return acc.item(), acc5.item()
            
        
    def attack_step(self, iden):
        
        config = self.config
        
        iden = iden.cpu()
        targets = iden.numpy().tolist()
        
        print('create initial w')
        w = self._create_initial_ws(targets)
        iden = torch.repeat_interleave(iden, config.num_candidate)
        
        w_optimized = []
        
        print(f'start optimize')
        
        for i in range(math.ceil(w.shape[0] / self.batch_size)):
            start_idx = i * self.batch_size
            end_idx = min((i+1) * self.batch_size, len(w))
            w_batch = w[start_idx: end_idx].to(config.device)
            iden_batch = iden[start_idx: end_idx].to(config.device)
            
            w_batch_optimized = self._optimize(w_batch, iden_batch).cpu()
            torch.cuda.empty_cache()
            
            w_optimized.append(w_batch_optimized)
        
        w_optimized = torch.cat(w_optimized, dim=0)
        
        # final selection
        
        eval_accs = []
        eval_acc5s = []
        
        print(f'fiinal selection')
        
        with torch.no_grad():
            candidates = self._generate_result_image(w_optimized)
            target_candidates = torch.chunk(candidates, chunks=len(targets), dim=0)
            
            for target, target_candidate, target_w in zip(targets, target_candidates, w_optimized):
                scores = 0
                for i in tqdm(range(config.final_selection_iters)):
                    scores += self._generate_label_scores(target_candidate, target, self.T, self.config.final_select_transform)
                top_scores, top_indice = torch.topk(scores, config.samples_per_targets)
                top_candidate = target_candidate[top_indice]
        
                self.folder_manager.save_result_images(top_candidate, target)
                
                top_candidate = top_candidate.to(config.device)
                
                T_acc, T_acc5 = self._calc_acc(top_candidate, target, self.T)
                E_acc, E_acc5 = self._calc_acc(top_candidate, target, self.E)
                
                eval_accs.append(E_acc)
                eval_acc5s.append(E_acc5)
                
                print(f'>> label {target}')
                print(f'target acc: {T_acc} acc5: {T_acc5}')
                print(f'eval acc: {E_acc} acc5: {E_acc5}')
                
        acc = np.array(eval_accs).mean()
        acc5 = np.array(eval_acc5s).mean()
        return {
            'acc': acc,
            'acc5': acc5
        }