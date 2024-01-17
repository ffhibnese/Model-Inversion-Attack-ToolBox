
import math
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

from ..base import BaseAttacker
from .config import RLBMIAttackConfig
from ..GMI.code.generator import Generator
from .code.agent import Agent

class RLBMIAttacker(BaseAttacker):
    
    def __init__(self, config: RLBMIAttackConfig) -> None:
        self._tag = f'{config.dataset_name}_{config.target_name}_{config.gan_dataset_name}'
        super().__init__(config)
        
        
        
    def get_tag(self) -> str:
        return self._tag
    
    def prepare_attack(self):
        config: RLBMIAttackConfig = self.config
        self.G = Generator(100).to(config.device)
        # self.D = DGWGAN(3).to(config.device)
        
        self.folder_manager.load_state_dict(self.G, 
                                   ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_G.tar'],
                                   device=config.device)
        # self.folder_manager.load_state_dict(self.D, 
        #                            ['GMI', f'{config.gan_dataset_name}_VGG16_GMI_D.tar'],
        #                            device=config.device)
    
    def attack_step(self, iden) -> dict:
        iden = iden.cpu().numpy().reshape(-1).tolist()
        config: RLBMIAttackConfig = self.config
        
        total = 0
        cnt = 0
        cnt5 = 0
        
        for i in iden:
            agent = Agent(state_size=config.z_dim, action_size=config.z_dim, device=config.device, hidden_size=256, action_prior="uniform")
            fake = self._inversion(agent, i)
            fake = fake.to(config.device)
            output = self.E(fake).result[0] # batch size = 1
            top_idx = torch.argmax(output)
            _, top5_idx = torch.topk(output, 5)

        total += 1
        if top_idx == i:
            cnt += 1
        if i in top5_idx:
            cnt5 += 1

        acc = cnt / total
        acc5 = cnt5 / total
        return {
            'acc': acc,
            'acc5': acc5
        }
            
        
    def _inversion(self, agent: Agent, label: int):
        best_score = 0
        
        
        self.G.eval()
        
        config: RLBMIAttackConfig = self.config
        device = config.device
        alpha = config.alpha
        
        for iter_time in tqdm(range(config.iter_times)):
            y = torch.LongTensor([label]).to(device)
            
            z = torch.randn((1, config.z_dim)).to(device)
            
            state = deepcopy(z.cpu().numpy())
            
            for t in range(config.max_step):
                action = agent.act(state)
                z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).to(device)
                next_state = deepcopy(z.cpu().numpy())
                state_image = self.G(z).detach()
                action_image = self.G(action.clone().detach().reshape((1, len(action))).to(device)).detach()
                
                state_output = self.T(state_image).result
                action_output = self.T(action_image).result
                score1 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
                score2 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_output, dim=-1)).data, 1, y))))
                score3 = math.log(max(1e-7, float(torch.index_select(F.softmax(state_output, dim=-1).data, 1, y)) - float(torch.max(torch.cat((F.softmax(state_output, dim=-1)[0,:y],F.softmax(state_output, dim=-1)[0,y+1:])), dim=-1)[0])))
                reward = 2 * score1 + 2 * score2 + 8 * score3
                
                
                done = t == config.max_step - 1
                
                agent.step(state, action, reward, next_state, done, t)
                state = next_state
            
            # Save the image with the maximum confidence score.
            test_images = []
            test_scores = []
            for i in range(1):
                with torch.no_grad():
                    z_test = torch.randn(1, config.z_dim).cuda()
                    for t in range(config.max_step):
                        state_test = z_test.cpu().numpy()
                        action_test = agent.act(state_test)
                        z_test = alpha * z_test + (1 - alpha) * action_test.clone().detach().reshape((1, len(action_test))).cuda()
                    test_image = self.G(z_test).detach()
                    test_images.append(test_image.cpu())
                    test_output = self.T(test_image).result
                    test_score = float(torch.mean(torch.diag(torch.index_select(F.softmax(test_output, dim=-1).data, 1, y))))
                test_scores.append(test_score)
            mean_score = sum(test_scores) / len(test_scores)
            
            if mean_score >= best_score:
                best_score = mean_score
                best_images = torch.vstack(test_images)
                self.folder_manager.save_result_image(best_images, label)
        return best_images