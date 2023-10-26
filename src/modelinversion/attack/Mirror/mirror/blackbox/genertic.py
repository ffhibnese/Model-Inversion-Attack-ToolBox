import torch
from dataclasses import dataclass
import torch.nn.functional as F
import torchvision.utils as vutils
from .blackbox_args import MirrorBlackBoxArgs
import random
import glob
import os
from torch import nn
from ...utils.img_utils import *
from ...mirror.select_w import find_closest_latent
from types import FunctionType

@dataclass
class VectorizedPopulation:
    population: int
    fitness_scores: float
    mutation_prob: float
    mutation_ce: float
    apply_noise_func: FunctionType
    clip_func: FunctionType
    compute_fitness_func: FunctionType
    bs: int
    embed_dim = 512
    device: str
    
    def compute_fitness(self):
        with torch.no_grad():
            bs = self.bs
            scores = []
            for i in range(0, len(self.population), bs):
                torch.cuda.empty_cache()
                data = self.population[i:min(len(self.population), i+bs)]
                # print(f'>>>> data shape: {data.shape}')
                scores.append(self.compute_fitness_func(data.to(self.device)).cpu())
                data.to('cpu')
            self.fitness_scores = torch.cat(scores, dim=0)
            
            assert self.fitness_scores.ndim == 1 and self.fitness_scores.shape[0] == len(self.population)
        
    def find_elite(self):
        with torch.no_grad():
            self.fitness_scores, indices = torch.sort(self.fitness_scores, 0, True)
            
            self.population = self.population[indices]
            
            return self.population[0].detach().clone(), self.fitness_scores[0].item()
    
    def _get_parents(self, k):
        with torch.no_grad():
            weights = torch.nn.functional.softmax(self.fitness_scores, dim=0).tolist()
            parents_idx = random.choices(list(range(len(weights))), weights=weights, k=2*k)
            
            return parents_idx[:k], parents_idx[k:]
    
    def _cross_over(self, parents1_idx, parents2_idx):
        with torch.no_grad():
            parents1_fitness_scores = self.fitness_scores[parents1_idx]
            parents2_fitness_scores = self.fitness_scores[parents2_idx]
            
            p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores)).unsqueeze(1)  # size: k, 1
            
            parents1 = self.population[parents1_idx].detach().clone()  # size: N, 512
            parents2 = self.population[parents2_idx].detach().clone()  # size: N, 512
            
            # print(f'>>>> device: population: {self.population.device}\t fit score: {self.fitness_scores.device}\t p: {p.device}')
            
            mask = torch.rand_like(parents1) < p
            
            return torch.where(mask, parents1, parents2)
    
    
    def _mutate(self, children):
        with torch.no_grad():
            mask = torch.rand_like(children) < self.mutation_prob
            
            children = self.apply_noise_func(children, mask, self.mutation_ce)
            
            return self.clip_func(children)
    
    def produce_next_generation(self, elite_value):
        with torch.no_grad():
            parents_tuple = self._get_parents(len(self.population)-1)
            children = self._cross_over(*parents_tuple)
            mutated_children = self._mutate(children)
            
            self.population = torch.cat([elite_value.unsqueeze(0), mutated_children], dim=0)
            self.compute_fitness()
        
    def visualize_imgs(self, file_dir, generate_images_func, k=8):
        # ws = self.population[:k]
        for i in range(k):
            out = generate_images_func(self.population[[i]], raw_img=True)
            vutils.save_image(out, os.path.join(file_dir, f'{i}.png'))
        
        
def init_population(args: MirrorBlackBoxArgs, target_label, target_model, compute_fitness_func):
    
    w_dir = os.path.join(args.pre_sample_dir, 'w')
    
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    all_ws_gen_files = sorted(glob.glob(os.path.join(w_dir, 'sample_*.pt')))
    
    all_w_mins_ls = []
    all_w_maxs_ls = []
    
    for ws_file in all_ws_gen_files:
    
        all_ws = torch.load(ws_file).detach()
        # print(f'all_ws.shape: {all_ws.shape}')
        all_ps = invert_lrelu(all_ws)
        all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
        all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
        all_p_mins = all_p_means - args.p_std_ce * all_p_stds
        all_p_maxs = all_p_means + args.p_std_ce * all_p_stds
        all_w_mins = lrelu(all_p_mins)
        all_w_maxs = lrelu(all_p_maxs)
        all_w_mins_ls.append(all_w_mins)
        all_w_maxs_ls.append(all_w_maxs)
    all_w_mins = torch.mean(torch.cat(all_w_mins_ls, dim=0)).cpu()
    all_w_maxs = torch.mean(torch.cat(all_w_maxs_ls, dim=0)).cpu()
    
    
    def clip_func(w):
        assert w.ndim == 2
        return clip_quantile_bound(w, all_w_mins, all_w_maxs)

    def apply_noise_func(w, mask, ce):
        assert w.ndim == 2
        p = invert_lrelu(w)
        noise = ((2*all_p_stds) * torch.rand_like(all_p_stds) - all_p_stds).cpu()
        noise = ce * noise
        p = p + mask*noise
        w = lrelu(p)
        return w
    
    select_w, conf = find_closest_latent(target_model, args.resolution, [target_label], k=args.population, arch_name=args.arch_name, pre_sample_dir=args.pre_sample_dir, bs=args.batch_size)
    
    select_w = select_w[target_label].cpu()
    conf = conf[target_label].cpu()
    
    return VectorizedPopulation(population=select_w, fitness_scores=conf, mutation_prob=args.mutation_prob, mutation_ce=args.mutation_ce, apply_noise_func=apply_noise_func, clip_func=clip_func, compute_fitness_func=compute_fitness_func, bs=args.batch_size, device=args.device)


from tqdm import tqdm
import math

def genetic_alogrithm(args: MirrorBlackBoxArgs, generate_images_func, target_label, target_model, compute_fitness_func):
    
    population = init_population(args, target_label=target_label, target_model=target_model, compute_fitness_func=compute_fitness_func)
    
    generations = args.generation
    
    for gen in tqdm(range(generations)):
        elite, elite_score = population.find_elite()
        # if elite_score >= math.log(args.min_score):
        #     population.visualize_imgs(os.path.join(args.work_dir, f'{target_label}/{gen}.png'), generate_images_func)
        #     return elite, elite_score
        
        population.produce_next_generation(elite)
    
    elite, elite_score = population.find_elite()
    population.visualize_imgs(os.path.join(args.result_dir, f'{target_label}'), generate_images_func)
    return elite, elite_score
    





