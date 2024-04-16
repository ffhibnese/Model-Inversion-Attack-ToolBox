import os
import importlib
from dataclasses import field, dataclass
from abc import abstractmethod, ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Iterator, Tuple, Callable, Sequence
import math

import torch
from torch import nn, Tensor, LongTensor
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm

from ...models import BaseImageClassifier
from ...utils import (
    unwrapped_parallel_module,
    ClassificationLoss,
    obj_to_yaml,
    print_as_yaml,
    DictAccumulator,
)


@dataclass
class BaseTrainConfig:

    experiment_dir: str
    save_name: str
    device: torch.device

    model: BaseImageClassifier
    optimizer: Optimizer
    lr_scheduler: Optional[LRScheduler] = None
    clip_grad_norm: Optional[float] = None

    save_per_epochs: int = 10


class BaseTrainer(ABC):

    def __init__(self, config: BaseTrainConfig, *args, **kwargs) -> None:
        self.config = config
        os.makedirs(config.experiment_dir, exist_ok=True)
        self.save_path = os.path.join(config.experiment_dir, config.save_name)

        self._epoch = 0
        self._iteration = 0

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def model(self):
        return self.config.model

    @property
    def optimizer(self):
        return self.config.optimizer

    @property
    def lr_scheduler(self):
        return self.config.lr_scheduler

    @abstractmethod
    def calc_loss(self, inputs, result, labels: torch.LongTensor):
        raise NotImplementedError()

    @torch.no_grad()
    def calc_acc(self, inputs, result, labels: torch.LongTensor):
        res = result[0]
        if isinstance(res, InceptionOutputs):
            res, _ = res
        assert res.ndim <= 2

        pred = torch.argmax(res, dim=-1)
        # print((pred == labels).float())
        return (pred == labels).float().mean()

    def calc_train_acc(self, inputs, result, labels: torch.LongTensor):
        return self.calc_acc(inputs, result, labels)

    def calc_test_acc(self, inputs, result, labels):
        return self.calc_acc(inputs, result, labels)

    def _update_step(self, loss):
        self.optimizer.zero_grad()
        if self.config.clip_grad_norm is not None:
            clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_grad_norm
            )
        loss.backward()
        self.optimizer.step()

    def prepare_input_label(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.config.device)
        labels = labels.to(self.config.device)

        return imgs, labels

    def _train_step(self, inputs, labels) -> OrderedDict:

        self.before_train_step()

        result = self.model(inputs)

        loss = self.calc_loss(inputs, result, labels)
        acc = self.calc_train_acc(inputs, result, labels)
        self._update_step(loss)

        return OrderedDict(loss=loss, acc=acc)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_step(self):
        self.model.train()

    def before_test_step(self):
        self.model.eval()

    def _train_loop(self, dataloader: DataLoader):

        self.before_train()

        accumulator = DictAccumulator()

        # iter_times = 0
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            # iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._train_step(inputs, labels)
            accumulator.add(step_res)

        self.after_train()

        return accumulator.avg()

    @torch.no_grad()
    def _test_step(self, inputs, labels):
        # self.model.eval()
        self.before_test_step()

        result = self.model(inputs)

        acc = self.calc_test_acc(inputs, result, labels)

        return OrderedDict(acc=acc)

    @torch.no_grad()
    def _test_loop(self, dataloader: DataLoader):

        accumulator = DictAccumulator()

        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._test_step(inputs, labels)
            accumulator.add(step_res)

        return accumulator.avg()

    def train(
        self, epoch_num: int, trainloader: DataLoader, testloader: DataLoader = None
    ):

        epochs = range(epoch_num)

        bestacc = 0
        bestckpt = None

        for epoch in epochs:

            self._epoch = epoch

            train_res = self._train_loop(trainloader)
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})

            if testloader is not None:
                test_res = self._test_loop(testloader)
                if 'acc' in test_res and test_res['acc'] > bestacc:
                    bestckpt = deepcopy(self.model).cpu().eval()
                print_as_yaml({'test': test_res})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % self.config.save_per_epochs == 0:
                self.save_state_dict()

        if bestckpt is None:
            self.save_state_dict()
        else:
            self.save_state_dict(bestckpt)

    def save_state_dict(self, model=None):
        if model is None:
            model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        torch.save({'state_dict': model.state_dict()}, self.save_path)


@dataclass
class SimpleTrainConfig(BaseTrainConfig):

    loss_fn: str | Callable = 'cross_entropy'


class SimpleTrainer(BaseTrainer):

    def __init__(self, config: SimpleTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.loss_fn = ClassificationLoss(config.loss_fn)

    def calc_loss(self, inputs, result, labels: LongTensor):
        result = result[0]
        if isinstance(result, InceptionOutputs):
            output, aux = result
            return self.loss_fn(output, labels) + self.loss_fn(aux, labels)
        return self.loss_fn(result, labels)


@dataclass
class DpsgdTrainConfig(SimpleTrainConfig):

    noise_multiplier: float = 0.01
    microbatch_size: int = 1


class DpsgdTrainer(SimpleTrainer):

    def __init__(self, config: DpsgdTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: DpsgdTrainConfig

    def _update_step(self, loss):
        bs = len(loss)

        parameters = [param for param in self.model.parameters() if param.requires_grad]

        grad = [torch.zeros_like(param) for param in parameters]
        num_microbatch = (bs - 1) // self.config.microbatch_size + 1

        max_norm = self.config.clip_grad_norm
        for j in range(0, bs, self.config.microbatch_size):
            self.optimizer.zero_grad()
            torch.autograd.backward(
                torch.mean(loss[j : min(j + self.config.microbatch_size, bs)]),
                retain_graph=True,
            )

            l2norm = 0.0
            for param in parameters:
                l2norm += (param.grad * param.grad).sum()
            l2norm = torch.sqrt(l2norm)

            coef = 1 if max_norm is None else (max_norm / max(max_norm, l2norm.item()))
            grad = [g + param.grad * coef for param, g in zip(parameters, grad)]

        if max_norm is None:
            max_norm = 1.0

        for param, g in zip(parameters, grad):
            param.grad.data = g
            if self.config.noise_multiplier > 0:
                param.grad.data += (
                    torch.randn_like(g.size())
                    * self.config.noise_multiplier
                    * float(max_norm)
                )
            param.grad.data /= num_microbatch

        self.optimizer.step()


@dataclass
class VibTrainConfig(SimpleTrainConfig):

    beta: float = 1e-2


class VibTrainer(SimpleTrainer):

    def __init__(self, config: BaseTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

    def calc_loss(self, inputs, result, labels: LongTensor):
        result, addition_info = result
        if isinstance(result, InceptionOutputs):
            output, aux = result
            main_loss = self.loss_fn(output, labels) + self.loss_fn(aux, labels)
        else:
            main_loss = self.loss_fn(result, labels)

        mu = addition_info['mu']
        std = addition_info['std']
        info_loss = (
            -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        )
        loss = main_loss + self.config.beta * info_loss
        return loss
