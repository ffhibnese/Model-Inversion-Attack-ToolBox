import os
import time
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import kornia
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tv_trans
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
from modelinversion.attack.base import BaseGANTrainArgs
from modelinversion.foldermanager import FolderManager

from modelinversion.metrics.base_od import DataLoader, FolderManager


from ..base import BaseGANTrainArgs, BaseGANTrainer
from ...models import *
from .code.generator import Generator
from .code.discri import MinibatchDiscriminator


@dataclass
class KedmiGANTrainArgs(BaseGANTrainArgs):

    arget_name: str = 'vgg16'

    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.9
    z_dim = 100


class KedmiGANTrainer(BaseGANTrainer):

    def __init__(
        self, args: BaseGANTrainArgs, folder_manager: FolderManager, **kwargs
    ) -> None:
        super().__init__(args, folder_manager, **kwargs)
        self.args: KedmiGANTrainArgs

    def get_tag(self) -> str:
        args = self.args
        return f'kedmi_{args.dataset_name}'

    def get_method_name(self) -> str:
        return 'KEDMI'

    def prepare_training(self):
        args = self.args
        self.G = Generator(args.z_dim).to(args.device)
        self.D = MinibatchDiscriminator().to(args.device)

        self.optim_G = torch.optim.Adam(
            self.G.parameters(), args.lr, (args.beta1, args.beta2)
        )
        self.optim_D = torch.optim.Adam(
            self.D.parameters(), args.lr, (args.beta1, args.beta2)
        )

        self.T = get_model(
            args.target_name,
            args.dataset_name,
            device=args.device,
            backbone_pretrain=False,
            defense_type=args.defense_type,
        )
        self.folder_manager.load_target_model_state_dict(
            self.T,
            args.dataset_name,
            args.target_name,
            device=args.device,
            defense_type=args.defense_type,
        )
        self.T.eval()

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor):
        shape = [len(real)] + [1] * (real.ndim - 1)
        alpha = torch.randn(shape).to(self.args.device)
        z = real + alpha * (fake - real)
        z = z.detach().clone()
        z.requires_grad_(True)

        o = self.D(z)
        g = torch.autograd.grad(o, z, create_graph=True)[0].view(len(real), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
        return gp

    def softXEnt(self, input, target):
        targetprobs = nn.functional.softmax(target, dim=1)
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(targetprobs * logprobs).sum() / input.shape[0]

    def train_dis_step(self, batch) -> OrderedDict:
        args = self.args

        batch = batch[0].to(args.device)
        bs = len(batch)
        z = torch.randn(bs, args.z_dim).to(args.device)

        fake = self.G(z)
        y_prob_pred = self.T(fake).result
        y_pred = torch.argmax(y_prob_pred, dim=-1).view(-1)

        _, dis_fake = self.D(fake)
        _, dis_real = self.D(batch)

        label_loss = self.softXEnt(dis_real, y_prob_pred)

        logsumexp_dis_real = torch.logsumexp(dis_real, dim=-1)
        unlabel_loss = 0.5 * (
            torch.mean(F.softplus(logsumexp_dis_real) - logsumexp_dis_real)
            + torch.mean(F.softplus(torch.logsumexp(dis_fake)))
        )

        loss = label_loss + unlabel_loss

        super().loss_update(loss, self.optim_D)

        acc = torch.mean((torch.argmax(dis_real, dim=-1) == y_pred).float)

        return OrderedDict(
            label_loss=label_loss, unlabel_loss=unlabel_loss, loss=loss, acc=acc
        )

    def train_gen_step(self, batch) -> OrderedDict:
        args = self.args
        batch = batch[0].to(args.device)
        bs = len(batch)
        z = torch.randn(bs, args.z_dim).to(args.device)

        fake = self.G(z)
        feature_fake, dis_fake = self.D(fake)
        feature_real, _ = self.D(batch)

        feature_fake = feature_fake.mean(dim=0)
        feature_real = feature_real.mean(dim=0)

        feature_loss = torch.mean((feature_fake - feature_real).abs())

        entropy_loss = -(
            F.softmax(dis_fake, dim=-1) * F.log_softmax(dis_fake, dim=-1)
        ).sum()

        loss = feature_loss + 1e-4 * entropy_loss
        super().loss_update(loss, self.optim_D)

        return OrderedDict(
            feature_loss=feature_loss, entropy_loss=entropy_loss, loss=loss
        )
