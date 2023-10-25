import argparse
import glob
import os
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

from facenet_pytorch import InceptionResnetV1
from attack.Mirror.utils.img_utils import normalize, clip_quantile_bound,  crop_and_resize, get_input_resolution
from attack.Mirror.mirror.classifiers.build_classifier import get_model
from dataclasses import dataclass
from attack.Mirror.genforce.get_genforce import get_genforce
from .genertic import genetic_alogrithm

from .blackbox_args import MirrorBlackBoxArgs

from metrics.knn import get_knn_dist

from .....utils import Tee
    
def mirror_blackbox_attack(
    # is_train,
    population,
    arch_name:str,
    eval_name: str,
    genforce_model_name: str,
    genforce_checkpoint_dir: str,
    result_dir,
    pre_sample_dir,
    target_labels: list,
    work_dir: str,
    classifiers_checkpoint_dir: str,
    dataset_name: str,
    batch_size : int,
    use_cache : bool,
    calc_knn : bool=False,
    device = 'cuda'
):
    
    target_net = get_model(arch_name, device, classifier_dir=classifiers_checkpoint_dir, dataset_name=dataset_name)
    resolution = get_input_resolution(arch_name)
    
    eval_net = get_model(arch_name, device, classifier_dir=classifiers_checkpoint_dir, dataset_name=dataset_name)
    eval_resolution = get_input_resolution(arch_name)
    
    args = MirrorBlackBoxArgs(
        population=population,
        arch_name=arch_name,
        eval_name=eval_name,
        genforce_model_name=genforce_model_name,
        genforce_checkpoint_dir=genforce_checkpoint_dir,
        result_dir=result_dir,
        target_labels=target_labels,
        work_dir=work_dir,
        classifiers_checkpoint_dir=classifiers_checkpoint_dir,
        batch_size=batch_size,
        device=device,
        resolution=resolution,
        eval_resolution=eval_resolution,
        use_cache=use_cache,
        pre_sample_dir=pre_sample_dir,
        calc_knn = calc_knn
    )
    
    
    generator, _ = get_genforce(genforce_model_name, device, genforce_checkpoint_dir, use_discri=False, use_w_space=args.use_w_space, use_z_plus_space=False, repeat_w=args.repeat_w)
    
    run(args, generator, target_net, eval_net)
    
def run(
    args: MirrorBlackBoxArgs,
    generator,
    target_net,
    eval_net
):
    def generate_images_func(w, raw_img=False):
        assert w.ndim == 2
        if raw_img:
            return generator(w.to(args.device))
        img = crop_and_resize(generator(w.to(args.device)), args.arch_name, args.resolution)
        return img
    
        
    print('--- train')

    for target in args.target_labels:
        
        os.makedirs(os.path.join(args.work_dir, f'{target}'), exist_ok=True)
        
        def compute_fitness_func(w):
            img = generate_images_func(w)
            assert img.ndim == 4
            # TODO: output
            pred = F.log_softmax(target_net(normalize(img*255., args.arch_name)).result, dim=1)
            score = pred[:, target]
            return score
        
        elite, elite_score = genetic_alogrithm(args, generate_images_func, target, target_model=target_net, compute_fitness_func=compute_fitness_func)
        
        score = math.exp(elite_score)
        print(f'target: {target} score: {score}')
        torch.save([elite, elite_score], os.path.join(args.work_dir, f'{target}', 'final.pt'))
        
    
    print('--- test')
    
    
    ws = []
    confs = []
    
    for target in tqdm(args.target_labels):
        path = os.path.join(args.work_dir, f'{target}', 'final.pt')
        if not os.path.exists(path):
            raise RuntimeError('the target label is not trained')
        
        w, score = torch.load(path)
        ws.append(w.to(args.device))
        confs.append(math.exp(score))
        
    ws = torch.stack(ws)
    imgs = generate_images_func(ws, raw_img=True)
    compute_conf(eval_net, args.eval_name, resolution=args.eval_resolution, targets=args.target_labels, imgs=imgs)
    # vutils.save_image(imgs, f'a.png', nrow=1)
    
    if args.calc_knn:
        
        feat_dir = os.path.join(args.classifiers_checkpoint_dir, "PLGMI", "celeba_private_feats")
        knn_dist = get_knn_dist(target_net, args.result_dir, feat_dir, resolution=112, device=args.device)
        print(f"knn dist: {knn_dist}")
        
        
def compute_conf(net, arch_name, resolution, targets, imgs):
    
    if arch_name == 'sphere20a':
        raise NotImplementedError('no support for sphere')
    
    # TODO: output
    logits = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name)).result.cpu()
    
    logits_softmax = F.softmax(logits, dim=1)
    
    
    k = 5
    print(f'top-{k} labels')
    
    topk_conf, topk_class = torch.topk(logits, k, dim=1)
    
    targets = torch.tensor(targets, dtype=torch.long)
    
    
    total_cnt = len(targets)
    topk_acc = (topk_class == targets.reshape(-1, 1)).sum() / total_cnt
    acc = (topk_class[:, 0] == targets).sum() / total_cnt
    # target_conf = logits_softmax[torch.arange(0, total_cnt, dtype=torch.long), targets]
    
    print(arch_name)
    print(f'top1 acc: {acc}')
    print(f'topk acc: {topk_acc}')
    
    
            
            