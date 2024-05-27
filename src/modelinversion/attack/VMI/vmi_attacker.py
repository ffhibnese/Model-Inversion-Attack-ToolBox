from ..attacker import *
from ..losses import VmiLoss
from ..optimize import BaseImageOptimizationConfig, MinerWhiteBoxOptimization
from typing import *


class VmiTrainer:
    def __init__(
        self,
        epochs: int,
        experiment_dir: str,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        flow_params: dict,
        device: torch.device,
        latents_mapping: Optional[Callable],
        classifier: BaseImageClassifier,
        loss_weights: dict,
        optimize_config: BaseImageOptimizationConfig
    ) -> None:
        self.epochs = epochs
        self.experiment_dir = experiment_dir

        self.input_size = input_size
        self.batch_size = batch_size
        self.generator = generator
        self.params = flow_params
        self.device = device
        self.mapping = latents_mapping

        self.classifier = classifier
        self.loss_weights = loss_weights
        self.optimize_config = optimize_config

    def init_flow_sampler(self):
        return LayeredFlowLatentsSampler(
            input_size=self.input_size,
            batch_size=self.batch_size,
            generator=self.generator,
            flow_params=self.params,
            device=self.device,
            latents_mapping=self.mapping,
            mode='train',
        )

    def init_loss_fn(self, miner: nn.Module):
        return VmiLoss(
            classifier=self.classifier,
            miner=miner,
            weights=self.loss_weights,
            batch_size=self.batch_size,
            device=self.device,
        )

    def init_optimization(
        self, config: BaseImageOptimizationConfig, loss_fn: BaseImageOptimization
    ):
        return MinerWhiteBoxOptimization(
            config=config, generator=self.generator, image_loss_fn=loss_fn
        )

    def train_single_miner(self, label: int):
        sampler = self.init_flow_sampler()
        loss_fn = self.init_loss_fn(sampler.miner)
        optimization = self.init_optimization(self.optimize_config, loss_fn)
        labels = label*torch.ones(self.batch_size).to(self.device).long()
        for epoch in range(self.epochs):
            optimization(sampler, labels)
        


class VmiAttacker(ImageClassifierAttacker):

    def __init__(self, config: ImageClassifierAttackConfig) -> None:
        super().__init__(config)

    def attack(self, target_list: list[int]):
        return super().attack(target_list)
