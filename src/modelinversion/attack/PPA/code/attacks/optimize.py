from losses.poincare import poincare_loss
import math

import numpy as np
import torch
import torch.nn as nn

class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']

    def optimize(self, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(
                    imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(
                outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                if torch.cuda.current_device() == 0:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f}'
                    )

        return w_batch.detach()

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

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
