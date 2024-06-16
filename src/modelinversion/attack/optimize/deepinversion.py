from collections import OrderedDict
from typing import Callable, Tuple
from torch import LongTensor, Tensor
from ...attack.optimize.base import SimpleWhiteBoxOptimizationConfig
from ...models import BaseImageGenerator
from ...utils import DeepInversionBNFeatureHook
from .base import *


class DeepInversionOptimizationConfig(SimpleWhiteBoxOptimizationConfig):

    pass


class DeepInversionOptimization(SimpleWhiteBoxOptimization):

    def __init__(
        self,
        config: SimpleWhiteBoxOptimizationConfig,
        image_loss_fn: Callable[
            [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        ],
    ) -> None:
        generator = lambda img, *args, **kwargs: img
        super().__init__(config, generator, image_loss_fn)
