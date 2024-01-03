import os
import statistics
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .config import BrepAttackConfig
from ..base import BaseAttacker
from ..KEDMI.code.generator import Generator
from ..KEDMI.code.discri import MinibatchDiscriminator



class BrepAttacker(BaseAttacker):
    
    def __init__(self, config: BrepAttackConfig) -> None:
        super().__init__(config)
        self.config: BrepAttackConfig
        
    def get_tag(self) -> str:
        config = self.config
        return f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}_{config.gan_target_name}'
    
    def prepare_attack(self):
        config = self.config
        
        self.G = Generator(config.z_dim).to(config.device)
        
        self.folder_manager.load_state_dict(self.G, 
                                   ['KEDMI', f'{config.gan_dataset_name}_{config.gan_target_name.upper()}_KEDMI_G.tar'],
                                   device=config.device)
        
        self.G.eval()
        
        self.init_z()
        
    def init_z(self):
        batch_size = self.batch_size
        target_labels = self.target_labels
        config = self.config
        
        num_idens = len(target_labels)
        initial_points = {}
        # current_iter = 0
        
        with torch.no_grad():
            # while True:
            for current_iter in tqdm(range(config.init_z_max_iter)):
                z = torch.randn(batch_size, config.z_dim).float().clamp(min=config.point_clamp_min, max=config.point_clamp_max).to(config.device)
                first_img = self.G(z)
                
                T_out = self.T(first_img).result
                T_pred = torch.argmax(T_out, dim=1)
                
                for i in range(batch_size):
                    current_label = T_pred[i].item()
                    if current_label in initial_points or current_label not in target_labels:
                        continue
                    
                    initial_points[current_label] = z[i]
                    
                # current_iter += 1
                if len(initial_points) == num_idens: # or current_iter > config.init_z_max_iter:
                    break
                
            unmap_labels = []
            for label in target_labels:
                if label not in initial_points.keys():
                    unmap_labels.append(label)
            print(f'labels {str(unmap_labels)} can not be generate in iter {config.init_z_max_iter}')
            
        self.initial_points = initial_points

    
    def attack_step(self, iden) -> dict:
        
        iden = iden.cpu().numpy().tolist()
        
        cnt = 0
        total = 0
        
        for label in iden:
            if label in self.initial_points:
                res = self.attack_single(label, self.initial_points[label])
                total += 1
                if res:
                    cnt += 1
        return {
            'acc': cnt / total
        }
                
    def _gen_points_on_sphere(self, current_point, points_count, sphere_radius, device):
    
        # get random perturbations
        points_shape = (points_count,) + current_point.shape
        perturbation_direction = torch.randn(*points_shape).to(device)
        dims = tuple([i for i in range(1, len(points_shape))])
        
        # normalize them such that they are uniformly distributed on a sphere with the given radius
        perturbation_direction = (sphere_radius/ torch.sqrt(torch.sum(perturbation_direction ** 2, axis = dims, keepdims = True))) * perturbation_direction
        
        # add the perturbations to the current point
        sphere_points = current_point + perturbation_direction
        return sphere_points, perturbation_direction
    
    def is_target_class(self, fake, target, model):
        val_iden = torch.argmax(model(fake).result, dim=1)
            
        val_iden[val_iden != target] = 0
        val_iden[val_iden == target] = 1
        return val_iden
        
    def attack_single(self, label, initial_z):
        current_iter = 0
        last_iter_when_radius_changed = 0
        
        config = self.config
        device = config.device
        
        current_z = initial_z.unsqueeze(0).to(device)
        
        current_sphere_radius = config.init_sphere_radius
    
        last_success_on_eval = False
        
        with torch.no_grad():
            # while current_iter - last_iter_when_radius_changed < config.max_iters_at_radius_before_terminate:
            for i in tqdm(range(config.max_iters_at_radius_before_terminate)):
                
                current_iter += 1
                
                new_radius = False
                
                step_size = min(current_sphere_radius / 3, 3)
                
                # sample points on the sphere
                new_points, perturbation_directions = self._gen_points_on_sphere(current_z[0], config.sphere_points_count, current_sphere_radius, device=device)
                
                # print(f">> {new_points.shape}")
                # exit()
                
                # get the predicted labels of the target model on the sphere points
                new_points_classification = self.is_target_class(self.G(new_points), label, self.T)
                
                if new_points_classification.sum() > 0.75 * config.sphere_points_count:
                    new_radius = True
                    last_iter_when_radius_changed = current_iter
                    current_sphere_radius *= config.sphere_expansion_coeff
                
                # get the update direction, which is the mean of all points outside boundary if 'repulsion_only' is used. Otherwise it is the mean of all points * their classification (1,-1)
                if config.repulsion_only == True:
                    new_points_classification = (new_points_classification - 1)/2
                    
                grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius
                
                # move the current point with stepsize towards grad_direction
                z_new = current_z + step_size * grad_direction
                z_new = z_new.clamp(min=config.point_clamp_min, max=config.point_clamp_max)
                
                current_img = self.G(z_new)
                
                if self.is_target_class(current_img, label, self.T)[0] == -1:
                    # log_file.write("current point is outside target class boundary")
                    continue
                
                current_z = z_new
                # eval_res = decision(current_img, E)[0].item()
                eval_res= torch.argmax(self.E(current_img).result, dim=1)[0].item()
                # print(eval_res)
                # exit()
                correct_on_eval = eval_res==label
                
                if new_radius:
                    # point_before_inc_radius = current_z.clone()
                    last_success_on_eval = correct_on_eval
                    continue
                
        self.folder_manager.save_result_image(current_img, label)
                
        return last_success_on_eval