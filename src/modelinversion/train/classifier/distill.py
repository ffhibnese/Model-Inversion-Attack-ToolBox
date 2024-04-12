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
            reduction='sum',
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
