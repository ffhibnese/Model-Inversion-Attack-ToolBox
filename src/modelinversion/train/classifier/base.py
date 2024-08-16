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
    print_split_line,
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
            if not isinstance(res, Tensor):
                res = res[0]
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

        # print(imgs.shape)
        # exit()

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

            # if i % 200 == 0:
            #     print_as_yaml({'epoch': self._epoch})
            #     print_as_yaml({'iter': i})
            #     print_as_yaml({'train': accumulator.avg()})
            #     accumulator.reset()

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

    def train_stop(
        self,
        epoch_num: int,
        trainloader: DataLoader,
        testloader: DataLoader = None,
        limit_accs: list[float] = [1.0],
        save_best_ckpts: bool = True,
    ):

        limit_idx = 0

        epochs = range(epoch_num)

        bestacc = 0
        bestckpt = None

        for epoch in epochs:

            self._epoch = epoch
            print_split_line()
            self.before_train()

            accumulator = DictAccumulator()

            # iter_times = 0
            for i, batch in enumerate(tqdm(trainloader, leave=False)):
                self._iteration = i
                # iter_times += 1
                inputs, labels = self.prepare_input_label(batch)
                step_res = self._train_step(inputs, labels)
                accumulator.add(step_res)

                if (i + 1) % (len(trainloader) // 10) == 0:
                    # print_as_yaml({'epoch': self._epoch})
                    # print_as_yaml({'iter': i})
                    # print_as_yaml({'train': accumulator.avg()})
                    if testloader is not None:
                        test_res = self._test_loop(testloader)
                        if 'acc' in test_res and test_res['acc'] > bestacc:
                            bestckpt = deepcopy(self.model).cpu().eval()
                            bestacc = test_res['acc']
                        print_as_yaml({'test': test_res})

                        if save_best_ckpts and bestacc >= limit_accs[limit_idx]:
                            limit_idx += 1
                            self.save_state_dict(
                                self.model, _save_subfolder=f'{bestacc:.4f}'
                            )
                            if limit_idx >= len(limit_accs):
                                break

            self.after_train()

            train_res = accumulator.avg()
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})

            if testloader is not None:
                test_res = self._test_loop(testloader)
                if 'acc' in test_res and test_res['acc'] > bestacc:
                    bestckpt = deepcopy(self.model).cpu().eval()
                    bestacc = test_res['acc']
                print_as_yaml({'test': test_res})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % self.config.save_per_epochs == 0:
                self.save_state_dict()

            # if bestacc >= limit_acc:
            #     break

        if bestckpt is None:
            self.save_state_dict()
        else:
            self.save_state_dict(bestckpt, test_acc=bestacc)

        print(f'best acc: {bestacc}')

    def save_state_dict(
        self,
        model=None,
        _save_subfolder=None,
        **kwargs,
    ):
        if model is None:
            model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        save_path = self.save_path
        if _save_subfolder is not None:
            save_path = os.path.join(self.config.experiment_dir, _save_subfolder)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, self.config.save_name)

        model.save_pretrained(save_path, **kwargs)

    def train(
        self,
        epoch_num: int,
        trainloader: DataLoader,
        testloader: DataLoader = None,
        limit_acc: float = 1.0,
        save_best_ckpts: bool = True,
    ):

        epochs = range(epoch_num)

        bestacc = 0
        bestckpt = None

        for epoch in epochs:

            self._epoch = epoch
            print_split_line()
            train_res = self._train_loop(trainloader)
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})

            if testloader is not None:
                test_res = self._test_loop(testloader)
                if save_best_ckpts and 'acc' in test_res and test_res['acc'] > bestacc:
                    bestckpt = deepcopy(self.model).cpu().eval()
                    bestacc = test_res['acc']
                print_as_yaml({'test': test_res})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % self.config.save_per_epochs == 0:
                self.save_state_dict()

            if bestacc >= limit_acc:
                break

        if bestckpt is None:
            self.save_state_dict()
        else:
            self.save_state_dict(bestckpt, test_acc=bestacc)

        print(f'best acc: {bestacc}')


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
            if not isinstance(output, Tensor):
                output = output[0]
            return self.loss_fn(output, labels) + self.loss_fn(aux, labels)
        return self.loss_fn(result, labels)
    
@dataclass
class MixTrainConfig(BaseTrainConfig):

    origin_loss_fn: str | Callable = 'cross_entropy'
    mix_loss_fn: str | Callable = 'cross_entropy'


class MixTrainer(BaseTrainer):

    def __init__(self, config: MixTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.origin_loss_fn = ClassificationLoss(config.origin_loss_fn)
        self.mix_loss_fn = ClassificationLoss(config.mix_loss_fn)

    def _apply_loss(self, inputs, targets, mask):
        mix_inputs = inputs[mask]
        mix_targets = targets[mask]
        ori_mask = ~mask
        ori_inputs = inputs[ori_mask]
        ori_targets = targets[ori_mask]
        mix_loss = self.mix_loss_fn(mix_inputs, mix_targets)
        ori_loss = self.origin_loss_fn(ori_inputs, ori_targets)
        return mix_loss + ori_loss

    def calc_loss(self, inputs, result, labels: LongTensor):
        mix_mask = labels[:, 1].bool()
        labels = labels[:, 0]
        result = result[0]
        if isinstance(result, InceptionOutputs):
            output, aux = result
            if not isinstance(output, Tensor):
                output = output[0]

            return self._apply_loss(output, labels, mix_mask) + self._apply_loss(aux, labels, mix_mask)
        return self._apply_loss(result, labels, mix_mask)
    
    def calc_acc(self, inputs, result, labels: LongTensor):
        if labels.ndim == 2:
            labels = labels[:, 0]
        return super().calc_acc(inputs, result, labels)


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


@dataclass
class BackdoorTrainConfig(SimpleTrainConfig):
    backdoor_resolution: int = 8
    backdoor_eps = 4 / 256


class BackdoorTrainer(SimpleTrainer):

    def __init__(self, config: BackdoorTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: BackdoorTrainConfig
        self.num_classes = unwrapped_parallel_module(self.config.model).num_classes
        self.label_backdoor = (
            (
                torch.randn(
                    (
                        self.num_classes,
                        3,
                        config.backdoor_resolution,
                        config.backdoor_resolution,
                    )
                )
                * config.backdoor_eps
            )
            .to(config.device)
            .detach()
            .requires_grad_(True)
        )

        self.label_backdoor_optimizer = type(config.optimizer)(
            [self.label_backdoor], lr=config.optimizer.defaults['lr']
        )

    def _update_step(self, loss):
        self.label_backdoor_optimizer.zero_grad()
        super()._update_step(loss)
        self.label_backdoor_optimizer.step()
        self.label_backdoor.data = torch.clamp(
            self.label_backdoor.data,
            -self.config.backdoor_eps,
            self.config.backdoor_eps,
        )

    def prepare_input_label(self, batch):
        """add backdoor label and noise"""
        inputs, labels = super().prepare_input_label(batch)
        backddoor_label = torch.randint_like(labels, 0, self.num_classes)
        backdoor_label = self.label_backdoor[backddoor_label]
        mask = torch.rand_like(labels, dtype=inputs.dtype) < 0.3
        inputs[mask][
            :, :, : self.config.backdoor_resolution, : self.config.backdoor_resolution
        ] = (
            inputs[mask][
                :,
                :,
                : self.config.backdoor_resolution,
                : self.config.backdoor_resolution,
            ]
            + backdoor_label[mask]
        )
        return inputs, labels


# class BaseTrainer(ABC):
