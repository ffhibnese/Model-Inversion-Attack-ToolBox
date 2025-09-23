import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .base import *
from ...models import BaseImageClassifier


@dataclass
class DistillTrainConfig(BaseTrainConfig):

    teacher: BaseImageClassifier = None


class DistillTrainer(BaseTrainer):

    def __init__(self, config: DistillTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.config: DistillTrainConfig

        if config.teacher is None:
            raise RuntimeError(f'Teacher model should not be None')

    def calc_loss(self, inputs, result, labels: LongTensor):
        result = result[0]
        teacher_result = self.config.teacher(inputs)[0]

        loss = F.kl_div(
            F.log_softmax(result, dim=-1),
            F.softmax(teacher_result, dim=-1),
            reduction='batchmean',
        )

        return loss

    @torch.no_grad()
    def calc_train_acc(self, inputs, result, labels: torch.LongTensor):
        res = result[0]
        if isinstance(res, InceptionOutputs):
            res, _ = res
        assert res.ndim <= 2

        teacher_result = self.config.teacher(inputs)[0]

        pred = torch.argmax(res, dim=-1)
        teacher_pred = torch.argmax(teacher_result, dim=-1)
        # print((pred == labels).float())
        return (pred == teacher_pred).float().mean()


@dataclass
class SmileTrainConfig(DistillTrainConfig):

    kl_coef: float = 1.0
    num_experts: int = 3


class SmileTrainer(DistillTrainer):

    def __init__(self, config: SmileTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.config: SmileTrainConfig

    def calc_loss(self, inputs, result, labels: LongTensor):

        result = result[0]
        teacher_result = self.config.teacher(inputs)[0]

        loss = 0.0

        for single_output in torch.chunk(result, self.config.num_experts, dim=0):
            kl_loss = F.kl_div(
                F.log_softmax(single_output, dim=-1),
                F.softmax(teacher_result, dim=-1),
                reduction='batchmean',
            )
            ce_loss = F.cross_entropy(single_output, teacher_result.argmax(dim=-1))
            loss += self.config.kl_coef * kl_loss + ce_loss

        return loss

    @torch.no_grad()
    def calc_train_acc(self, inputs, result, labels: torch.LongTensor):
        res = result[0]
        if isinstance(res, InceptionOutputs):
            res, _ = res
        assert res.ndim <= 2

        teacher_result = self.config.teacher(inputs)[0]

        result = torch.chunk(res, self.config.num_experts, dim=0)
        # mean of result
        res = torch.mean(torch.stack(result, dim=0), dim=0)

        pred = torch.argmax(res, dim=-1)
        teacher_pred = torch.argmax(teacher_result, dim=-1)
        # print((pred == labels).float())
        return (pred == teacher_pred).float().mean()

    @torch.no_grad()
    def calc_acc(self, inputs, result, labels: torch.LongTensor):
        return self.calc_train_acc(inputs, result, labels)
