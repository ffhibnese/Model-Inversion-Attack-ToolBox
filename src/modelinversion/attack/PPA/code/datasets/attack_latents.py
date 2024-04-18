import numpy as np
import torch
from torch.utils.data import Dataset
import wandb
import yaml


class AttackLatents(Dataset):

    def __init__(self, attack_run_path=None, latent_file=None, transform=None):

        assert bool(attack_run_path) != bool(
            latent_file
        ), 'Either attack_run_path or latent_file must be specified'

        self.attack_run_path = attack_run_path
        self.latent_file = latent_file
        self.transform = transform

        if attack_run_path:
            weights_file_name = 'results/optimized_w_selected_' + attack_run_path.split(
                '/')[-1] + '.pt'
            w_optimized = wandb.restore(weights_file_name,
                                        run_path=attack_run_path,
                                        replace=True,
                                        root='wandb/downloads')
            self.latents = torch.load(w_optimized.name)

        api = wandb.Api()
        run = api.run(attack_run_path)
        self.target_identities = run.config['targets']
        if self.target_identities == 'all':
            self.num_classes = len(self.latents) // run.config['final_samples']
        else:
            self.num_classes = len(self.target_identities)
        samples_per_class = len(self.latents) // self.num_classes
        targets = [[i for j in range(samples_per_class)]
                   for i in range(self.num_classes)]
        self.targets = [t for sublist in targets for t in sublist]

        assert len(self.latents) == len(self.targets)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        if self.transform:
            latent = self.transform(latent)
        return latent, self.targets[idx]