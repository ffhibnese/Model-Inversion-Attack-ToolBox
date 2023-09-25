#!/usr/bin/env python3
# coding=utf-8
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

import numpy as np

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere
import ccs19_model_inversion
from my_utils import normalize, clip_quantile_bound, create_folder, Tee, add_conf_to_tensors, crop_and_resize
from my_datasets import IndexedDataset, StyleGANSampleDataSet
from my_target_models import get_model, get_input_resolution


random.seed(0)


class Sample:
    def __init__(self, value, fitness_score=-1):
        """
        value is a tensor
        """
        self.value = value
        self.fitness_score = fitness_score


class VectorizedPopulation:
    def __init__(self, population, fitness_scores, mutation_prob, mutation_ce, apply_noise_func, clip_func, compute_fitness_func):
        """
        population is a tensor with size N,512
        fitness_scores is a tensor with size N
        """
        self.population = population
        self.fitness_scores = fitness_scores
        self.mutation_prob = mutation_prob
        self.mutation_ce = mutation_ce
        self.apply_noise_func = apply_noise_func
        self.clip_func = clip_func
        self.compute_fitness_func = compute_fitness_func

    def compute_fitness(self):
        bs = 50
        scores = []
        for i in range(0, len(self.population), bs):
            data = self.population[i:i+bs]
            scores.append(self.compute_fitness_func(data))
        self.fitness_scores = torch.cat(scores, dim=0)
        assert self.fitness_scores.ndim == 1 and self.fitness_scores.shape[0] == len(self.population)

    def find_elite(self):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[0].detach().clone(), self.fitness_scores[0].item())

    def __get_parents(self, k):
        weights = F.softmax(self.fitness_scores, dim=0).tolist()
        parents_ind = random.choices(list(range(len(weights))), weights=weights, k=2*k)
        parents1_ind = parents_ind[:k]
        parents2_ind = parents_ind[k:]

        return parents1_ind, parents2_ind

    def __crossover(self, parents1_ind, parents2_ind):
        parents1_fitness_scores = self.fitness_scores[parents1_ind]
        parents2_fitness_scores = self.fitness_scores[parents2_ind]
        p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores)).unsqueeze(1)  # size: N, 1
        parents1 = self.population[parents1_ind].detach().clone()  # size: N, 512
        parents2 = self.population[parents2_ind].detach().clone()  # size: N, 512
        mask = torch.rand_like(parents1)
        mask = (mask < p).float()
        return mask*parents1 + (1.-mask)*parents2

    def __mutate(self, children):
        mask = torch.rand_like(children)
        mask = (mask < self.mutation_prob).float()
        children = self.apply_noise_func(children, mask, self.mutation_ce)
        return self.clip_func(children)

    def produce_next_generation(self, elite):
        parents1_ind, parents2_ind = self.__get_parents(len(self.population)-1)
        children = self.__crossover(parents1_ind, parents2_ind)
        mutated_children = self.__mutate(children)
        self.population = torch.cat((elite.value.unsqueeze(0), mutated_children), dim=0)
        self.compute_fitness()

    def visualize_imgs(self, filename, generate_images_func, k=8):
        ws = self.population[:k]
        out = generate_images_func(ws, raw_img=True)
        vutils.save_image(out, filename)


def init_population(args):
    """
    find args.n images with highest confidence
    """
    all_ws_pt_file = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt'

    # compute bound in p space
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)

    all_ws = torch.load(all_ws_pt_file).to(args.device)
    print(f'all_ws.shape: {all_ws.shape}')
    all_ps = invert_lrelu(all_ws)
    all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
    all_p_mins = all_p_means - args.p_std_ce * all_p_stds
    all_p_maxs = all_p_means + args.p_std_ce * all_p_stds
    all_w_mins = lrelu(all_p_mins)
    all_w_maxs = lrelu(all_p_maxs)
    print(f'all_w_mins.shape: {all_w_mins.shape}')

    def clip_func(w):
        assert w.ndim == 2
        return clip_quantile_bound(w, all_w_mins, all_w_maxs)

    def apply_noise_func(w, mask, ce):
        assert w.ndim == 2
        p = invert_lrelu(w)
        noise = (2*all_p_stds) * torch.rand_like(all_p_stds) - all_p_stds
        noise = ce * noise
        p = p + mask*noise
        w = lrelu(p)
        return w

    if args.use_cache:
        all_logits_file = os.path.join('./blackbox_attack_data/stylegan',
                                       args.arch_name,
                                       'use_dropout' if args.use_dropout else 'no_dropout',
                                       'all_logits.pt')
        all_logits = torch.load(all_logits_file).to(args.device)
        all_prediction = F.log_softmax(all_logits, dim=1)[:, args.target]
    else:
        img_dir = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8/'
        train_set = StyleGANSampleDataSet(all_ws_pt_file, img_dir, args.arch_name, args.resolution, end_index=100_000, preload=args.preload)
        kwargs = {'num_workers': 16, 'pin_memory': True} if not args.no_cuda else {}
        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, **kwargs)

        all_prediction = []
        for _, prediction, _ in train_loader:
            all_prediction.append(F.log_softmax(prediction.to(args.device), dim=1)[:, args.target])
        all_prediction = torch.cat(all_prediction, dim=0)
    print('all_prediction.shape', all_prediction.shape, 'device', all_prediction.device)
    topk_conf, topk_ind = torch.topk(all_prediction, args.population_size, dim=0, largest=True, sorted=True)
    # print(topk_conf)
    # print(topk_ind)
    population = all_ws[topk_ind].detach().clone()
    fitness_scores = topk_conf
    return VectorizedPopulation(population, fitness_scores, args.mutation_prob, args.mutation_ce, apply_noise_func, clip_func, args.compute_fitness_func)


def genetic_algorithm(args, generator, generate_images_func):
    population = init_population(args)
    generations = args.generations
    for gen in range(generations):
        elite = population.find_elite()
        print(f'elite at {gen}-th generation: {elite.fitness_score}')
        population.visualize_imgs(os.path.join(args.exp_name, f'{gen}.png'), generate_images_func)

        if elite.fitness_score >= math.log(args.min_score):  # fitness_score is log_softmax
            return elite
        population.produce_next_generation(elite)

    # return the elite
    elite = population.find_elite()
    population.visualize_imgs(os.path.join(args.exp_name, f'{gen+1}.png'), generate_images_func)
    return elite


def compute_conf(net, arch_name, resolution, targets, imgs):
    if arch_name == 'sphere20a':
        sphere20_theta_net = getattr(net_sphere, 'sphere20a')(use_theta=True)
        sphere20_theta_net.load_state_dict(torch.load('./sphere20a_20171020.pth'))
        sphere20_theta_net.to('cuda')

    try:
        label_logits_dict = torch.load(os.path.join('./centroid_data', arch_name, 'test/centroid_logits.pt'))
    except FileNotFoundError:
        print('Note: centroid_logits.pt is not found')
        label_logits_dict = None

    outputs = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name))
    if arch_name == 'sphere20a':
        outputs = outputs[0]
        # net.feature = True
        # logits = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name)).cpu()
        # net.feature = False
        logits = sphere20_theta_net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name)).cpu()
    else:
        logits = outputs.cpu()
    logits_softmax = F.softmax(outputs, dim=1)

    target_conf = []

    k = 5
    print(f'top-{k} labels')
    topk_conf, topk_class = torch.topk(outputs, k, dim=1)
    correct_cnt = 0
    topk_correct_cnt = 0
    total_cnt = len(targets)
    l2_dist = []
    for i in range(len(targets)):
        t = targets[i]
        target_conf.append(logits_softmax[i, t].item())
        if label_logits_dict is not None:
            l2_dist.append(torch.dist(logits[i], label_logits_dict[t]).item())
        if topk_class[i][0] == t:
            correct_cnt += 1
        if t in topk_class[i]:
            topk_correct_cnt += 1
    # print('target conf:', target_conf)
    l2_dist = l2_dist and sum(l2_dist)/len(l2_dist)
    print(arch_name)
    print(f'top1 acc: {correct_cnt}/{total_cnt} = {correct_cnt/total_cnt:.4f}')
    print(f'topk acc: {topk_correct_cnt}/{total_cnt} = {topk_correct_cnt/total_cnt:.4f}')
    print(f'l2_dist: {l2_dist:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--population_size', default=100, type=int, help='population size')
    parser.add_argument('--arch_name', default='ccs19ami_facescrub_rgb', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--log-interval', type=int, default=10, metavar='')
    parser.add_argument('--mutation_prob', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--mutation_ce', type=float, default=0.1, help='mutation coefficient')
    parser.add_argument('--generations', default=100, type=int, help='total generations')
    parser.add_argument('--target', default=0, type=int, help='the target label')
    parser.add_argument('--test_target', help='the only one target to test, or multiple targets separated by ,')
    parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std')
    parser.add_argument('--min_score', type=float, default=0.95, help='once reaching the score, terminate the attack')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--use_cache', action='store_true')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    args.exp_name = os.path.join('genetic_attack', args.exp_name)
    create_folder(args.exp_name)
    Tee(os.path.join(args.exp_name, 'output.log'), 'w')
    print(args)
    print(datetime.now())

    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device = device
    print(f'using device: {device}')

    net = get_model(args.arch_name, device)
    if args.arch_name == 'vgg16bn':
        args.test_arch_name = 'vgg16'
        result_dir = 'gpu_vggface_vgg16bn'
    elif args.arch_name == 'resnet50':
        args.test_arch_name = 'inception_resnetv1_vggface2'
        result_dir = 'gpu_vggface2_resnet50'
    elif args.arch_name == 'vgg16':
        args.test_arch_name = 'vgg16bn'
        result_dir = 'gpu_vggface_vgg16'
    elif args.arch_name == 'inception_resnetv1_vggface2':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_inceptionrnv1'
    elif args.arch_name == 'inception_resnetv1_casia':
        args.test_arch_name = 'sphere20a'
        result_dir = 'casia_inceptionrnv1'
    elif args.arch_name == 'sphere20a':
        args.test_arch_name = 'inception_resnetv1_casia'
        result_dir = 'gpu_vggface_sphere20a'
    else:
        raise AssertionError('wrong arch_name')

    args.resolution = get_input_resolution(args.arch_name)
    args.test_resolution = get_input_resolution(args.test_arch_name)

    use_w_space = True
    repeat_w = True  # if False, opt w+ space
    # num_layers = 14  # 14 for stylegan w+ space
    # num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024

    # genforce_model = 'pggan_celebahq1024'
    genforce_model = 'stylegan_celeba_partial256'
    # genforce_model = 'stylegan_celebahq1024'
    # genforce_model = 'stylegan2_ffhq1024'
    if not genforce_model.startswith('stylegan'):
        use_w_space = False

    def get_generator(batch_size, device):
        from genforce import my_get_GD
        use_discri = False
        generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w)
        return generator

    generator = get_generator(args.bs, device)

    def generate_images_func(w, raw_img=False):
        assert w.ndim == 2
        if raw_img:
            return generator(w.to(device))
        img = crop_and_resize(generator(w.to(device)), args.arch_name, args.resolution)
        return img

    def compute_fitness_func(w):
        img = generate_images_func(w)
        assert img.ndim == 4
        if args.arch_name == 'sphere20a':
            pred = F.log_softmax(net(normalize(img*255., args.arch_name))[0], dim=1)
            score = pred[:, args.target]
        else:
            pred = F.log_softmax(net(normalize(img*255., args.arch_name)), dim=1)
            score = pred[:, args.target]
        return score
    args.compute_fitness_func = compute_fitness_func

    if args.test_only:
        targets = list(map(int, args.test_target.split(',')))
        ws = []
        all_confs = []
        for t in targets:
            final_sample = torch.load(os.path.join('./genetic_attack', f'{result_dir}_{t}', 'final_w.pt'))
            w = final_sample.value
            ws.append(w.to(device))
            score = math.exp(final_sample.fitness_score)
            all_confs.append(score)
        ws = torch.stack(ws, dim=0)
        imgs = generate_images_func(ws, raw_img=True)
        compute_conf(net, args.arch_name, args.resolution, targets, imgs)
        compute_conf(get_model(args.test_arch_name, device), args.test_arch_name, args.test_resolution, targets, imgs)

        imgs = add_conf_to_tensors(imgs, all_confs)
        create_folder('./tmp')
        vutils.save_image(imgs, f'./tmp/all_{args.arch_name}_ge_images.png', nrow=1)
        return

    res = genetic_algorithm(args, generator, generate_images_func)
    score = math.exp(res.fitness_score)
    print(f'final confidence: {score}')
    torch.save(res, os.path.join(args.exp_name, 'final_w.pt'))
    print(datetime.now())


if __name__ == '__main__':
    with torch.no_grad():
        main()
