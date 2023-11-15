import torch
from torchvision.transforms import Resize
from torch.nn import functional as F
import numpy as np
from torchvision import utils as uts
from tqdm import tqdm


class Optimizer(object):
    def __init__(self, eva_model, generator, target_label, mask, nv=15, beta=0.04,
                 input_latent=True, face_shape=None, device='cuda', direct='gen_figures/ggsmd/'):
        if face_shape is None:
            face_shape = [160, 160]
        self.M = mask
        self.f = eva_model.eval().to(device)
        self.beta = beta
        self.nv = nv
        self.g = generator.eval().to(device)
        self.input_latent = input_latent
        self.trans = Resize(face_shape)
        self.target = target_label
        self.device = device
        self.dir    = direct

    def get_rand_w(self):
        noise = torch.randn(3, 512).to(self.device)
        with torch.no_grad():
            latent_r = self.g.style(noise).mean(0).cpu()
        return np.array(latent_r)

    def gen_img(self, x):
        x = torch.tensor(x).to(self.device)
        with torch.no_grad():
            imgs_gen, _ = self.g([x], input_is_latent=self.input_latent)
        return imgs_gen

    def save_img(self, img, best_id, only_best=False):
        file_name  = f'de_label{self.target}.jpg'
        file_name2 = f'de_label{self.target}_best.jpg'
        uts.save_image(
            img[best_id].unsqueeze(0),
            self.dir+file_name2,
            nrow=1,
            normalize=True,
            range=(-1, 1),
            )
        if not only_best:
            uts.save_image(
                img,
                self.dir+file_name,
                nrow=8,
                normalize=True,
                range=(-1, 1),
                )

    @staticmethod
    def resize_img_gen_shape(img, trans):
        t_img = trans(img)
        face_input = t_img.clamp(min=-1, max=1).add(1).div(2)
        return face_input

    def cal_fitness(self, x):
        with torch.no_grad():
            imgs_gen = self.gen_img(x)
            face_in = self.resize_img_gen_shape(imgs_gen, self.trans)
            before_no, _ = self.f.forward_feature(face_in)
            predicti = self.f.forward_classifier(before_no)
        prob = F.softmax(predicti, dim=1)
        fitness = torch.zeros_like(prob[:, 1])
        for i in range(fitness.shape[0]):
            topk, idx = torch.topk(prob[i,:], self.M)
            fitness[i] = prob[i, self.target] if self.target in idx else -topk.sum()
        return np.array(fitness.cpu())


class DE_c2b_5_bin(object):
    def __init__(self, optim, max_gen, x, popsize=32):
        x = np.array(x.detach().cpu())
        self.optim = optim
        self.max_gen = max_gen
        self.batch_size = x.shape[0]
        self.popsize = popsize
        self.pop = x
        if self.batch_size < self.popsize:
            self.extend()

    def extend(self):
        tmp = []
        num_lack = self.popsize - self.batch_size
        for _ in range(num_lack):
            idx = np.random.randint(0,self.batch_size, size=3)
            while idx[0] == idx[1] or idx[0] == idx[2] or idx[1] == idx[2]:
                idx = np.random.randint(0,self.batch_size, size=3)
            x1 = self.pop[idx[0]]
            x2 = self.pop[idx[1]]
            x3 = self.pop[idx[2]]
            u  = 0.5*((0.5*(x1+x2))+x3)
            rand_w = self.optim.get_rand_w()
            u  = 0.7*u + 0.3*rand_w
            tmp.append(np.expand_dims(u, axis=0))
        uu = np.concatenate(tmp, axis=0)
        self.pop = np.concatenate([self.pop, uu], axis=0)

    def step(self, if_cross=True):
        nv = self.optim.nv
        beta = self.optim.beta
        fitness_par = self.optim.cal_fitness(self.pop)
        max_id = np.argmax(fitness_par)
        x_best = self.pop[max_id]

        children = []
        for i in range(self.popsize):
            child = self.gen_one_child(beta, self.pop[i], x_best, nv)
            child = np.expand_dims(child, axis=0)
            children.append(child)
        children = np.concatenate(children, axis=0)
        fitness_chi = self.optim.cal_fitness(children)

        print_fit = []
        for i in range(self.popsize):
            if fitness_par[i] < fitness_chi[i]:
                pr = 0.2
                if not if_cross:
                    pr = 0.0
                # self.pop[i] = children[i]
                self.recombine(children[i], pr, i)
                print_fit.append(fitness_chi[i])
            else:
                pr = 0.8
                if not if_cross:
                    pr = 1.0
                self.recombine(children[i], pr, i)
                print_fit.append(fitness_par[i])
        return print_fit

    def recombine(self, child, pr, i):
        dim = child.shape[-1]
        for j in range(dim):
            if np.random.rand(1) > pr:
                self.pop[i,j] = child[j]

    def gen_one_child(self, beta, x_i, x_best, nv):
        tmp = x_i + 0.025 * (x_best - x_i)  # 0.025
        idx = np.random.randint(0, self.popsize, size=(nv,2))
        for i in range(nv):
            i2, i3 = idx[i,0], idx[i,1]
            tmp += beta*(self.pop[i2]-self.pop[i3])
        return tmp

    def run(self, disturb=0.0):
        pbar = tqdm(range(self.max_gen))
        for i in pbar:
            if i < self.max_gen-20:
                prob = self.step()
            else:
                prob = self.step(if_cross=False)
            # print(f'step {i}:\n{prob}')
            if i < self.max_gen-20:
                self.pop += np.random.randn(*self.pop.shape) * disturb
            pbar.set_description(
                (
                    f'label-{self.optim.target}; maxProb-{np.max(prob)}'
                )
            )
        print(f'label {self.optim.target}:\n{prob}')

    def get_img(self, num_imgs, only_best=False):
        imgs = self.optim.gen_img(self.pop[0:num_imgs])
        fitness_par = self.optim.cal_fitness(self.pop)
        best_id = np.argmax(fitness_par)
        self.optim.save_img(imgs, best_id, only_best)


class DE_c2b_5_bin2(object):
    def __init__(self, optim, max_gen, x, popsize=32):
        x = np.array(x.detach().cpu())
        self.optim = optim
        self.max_gen = max_gen
        self.batch_size = x.shape[0]
        self.popsize = popsize
        self.pop = x
        if self.batch_size < self.popsize:
            self.extend()

    def extend(self):
        tmp = []
        num_lack = self.popsize - self.batch_size
        for _ in range(num_lack):
            idx = np.random.randint(0,self.batch_size, size=3)
            while idx[0] == idx[1] or idx[0] == idx[2] or idx[1] == idx[2]:
                idx = np.random.randint(0,self.batch_size, size=3)
            x1 = self.pop[idx[0]]
            x2 = self.pop[idx[1]]
            x3 = self.pop[idx[2]]
            u  = 0.5*((0.5*(x1+x2))+x3)
            rand_w = self.optim.get_rand_w()
            u  = 0.7*u + 0.3*rand_w
            tmp.append(np.expand_dims(u, axis=0))
        uu = np.concatenate(tmp, axis=0)
        self.pop = np.concatenate([self.pop, uu], axis=0)

    def step(self, step_i):
        # nv = self.optim.nv
        # beta = self.optim.beta
        nv = 4
        if step_i < self.max_gen-30:
            beta = 0.1
        else:
            beta = 0.1 * (self.max_gen-step_i)/50
        fitness_par = self.optim.cal_fitness(self.pop)
        max_id = np.argmax(fitness_par)
        x_best = self.pop[max_id]

        children = []
        for i in range(self.popsize):
            child = self.gen_one_child(beta, self.pop[i], x_best, nv)
            child = np.expand_dims(child, axis=0)
            children.append(child)
        children = np.concatenate(children, axis=0)
        fi_children = self.cross(children)
        fitness_chi = self.optim.cal_fitness(fi_children)

        print_fit = []
        for i in range(self.popsize):
            if fitness_par[i] < fitness_chi[i]:
                self.pop[i] = fi_children[i]
                print_fit.append(fitness_chi[i])
            else:
                print_fit.append(fitness_par[i])
        return print_fit

    def cross(self, children, pr=0.5):
        batch, dim = children.shape
        fi_children = np.zeros_like(children)
        for i in range(batch):
            for j in range(dim):
                fi_children[i,j] = children[i,j] if np.random.rand(1) > pr else self.pop[i,j]
        return fi_children

    def gen_one_child(self, beta, x_i, x_best, nv):
        tmp = x_i + 0.02 * (x_best - x_i)  # facescrub: 0.04
        # idx = np.random.randint(0, self.popsize, size=(nv,2))
        idx = np.random.choice(self.popsize, int(nv*2)).reshape(nv, 2)
        for i in range(nv):
            i2, i3 = idx[i,0], idx[i,1]
            tmp += beta*(self.pop[i2]-self.pop[i3])
        return tmp

    # 执行max_gen代差分进化
    def run(self, disturb=0.0):
        pbar = tqdm(range(self.max_gen))
        for i in pbar:
            prob = self.step(i)
            # print(f'step {i}:\n{prob}')
            if i < self.max_gen-20:
                self.pop += np.random.randn(*self.pop.shape) * disturb
            pbar.set_description(
                (
                    f'label-{self.optim.target}; maxProb-{np.max(prob)}'
                )
            )
        print(f'label {self.optim.target}:\n{prob}')

    def get_img(self, num_imgs, only_best=False):
        imgs = self.optim.gen_img(self.pop[0:num_imgs])
        # fitness_par = self.optim.cal_fitness(self.pop)
        # best_id = np.argmax(fitness_par)
        # self.optim.save_img(imgs, best_id, only_best)
        return imgs
