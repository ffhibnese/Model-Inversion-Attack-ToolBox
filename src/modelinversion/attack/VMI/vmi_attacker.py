from ..attacker import *
from ..losses import VmiLoss
from ...utils import Logger
from ..optimize import BaseImageOptimizationConfig, MinerWhiteBoxOptimization
from ...sampler import FlowConfig
from typing import *
import time
import multiprocessing


class VmiTrainer:
    def __init__(
        self,
        epochs: int,
        experiment_dir: str,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        flow_params: FlowConfig,
        device: torch.device,
        latents_mapping: Optional[Callable],
        classifier: BaseImageClassifier,
        loss_weights: dict,
        optimize_config: BaseImageOptimizationConfig,
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

    def train_single_miner(self, args):
        label, root_path, img_path = args
        sampler = self.init_flow_sampler()
        loss_fn = self.init_loss_fn(sampler.miner)
        optimization = self.init_optimization(self.optimize_config, loss_fn)

        optimizer = None
        for epoch in range(self.epochs):
            output, optimizer = optimization(sampler, label, optimizer)

        # save miner
        label_path = os.path.join(root_path, str(label))
        safe_save(
            {'state_dict': sampler.miner.state_dict()},
            label_path,
            f'{label}_minor_{self.epochs}.pt',
        )

        # save images
        safe_save(
            output.images,
            img_path,
            f'training_samples_{self.optimize_config.generate_num}_{label}.pt',
        )
        safe_save(output.images[:5], img_path, f'training_samples_{5}_{label}.pt')

    def train_miners(self, cores: int, targets: list[int], root_path: str):
        # prepare logger
        now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
        logger = Logger(root_path, f'attack_{now_time}.log')

        img_path = os.path.join(root_path, 'samples')
        root_path = os.path.join(root_path, 'minors')
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(processes=cores) as pool:
            tasks = [(i, root_path, img_path) for i in targets]
            pool.map(self.train_single_miner, tasks)


class VmiAttacker(ImageClassifierAttacker):

    def __init__(self, config: ImageClassifierAttackConfig) -> None:
        super().__init__(config)

    def attack(self, target_list: list[int]):
        return super().attack(target_list)
