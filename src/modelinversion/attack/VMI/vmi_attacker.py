from ..attacker import *
from ..attacker import _ImageClassifierAttackerOptimizedOutput
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
        transform: nn.Module = None
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
        self.transform = transform

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

        for epoch in range(self.epochs):
            output = optimization(sampler, label)

        # save miner
        label_path = os.path.join(root_path, str(label))
        safe_save(
            sampler.miner.state_dict(),
            label_path,
            f'{label}_minor_{self.epochs}.pt',
        )

        # save images
        safe_save(
            output.images,
            img_path,
            f'{label}_training_samples_{self.optimize_config.generate_num}.pt',
        )
        safe_save(output.images[:5], img_path, f'{label}_training_samples_{5}.pt')

    def train_miners(self, cores: int, targets: list[int], root_path: str):
        img_path = os.path.join(root_path, 'samples')
        root_path = os.path.join(root_path, 'minors')
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(processes=cores) as pool:
            tasks = [(i, root_path, img_path) for i in targets]
            pool.map(self.train_single_miner, tasks)


class VmiAttacker:

    def __init__(
        self,
        epochs: int,
        eval_metrics: list[BaseImageMetric],
        experiment_dir: str,
        eval_bs: int,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        flow_params: FlowConfig,
        device: torch.device,
        latents_mapping: Optional[Callable],
    ) -> None:
        self.metrics = eval_metrics
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.eval_bs = eval_bs

        self.input_size = input_size
        self.batch_size = batch_size
        self.generator = generator
        self.params = flow_params
        self.device = device
        self.mapping = latents_mapping

    def trained_flow_sampler(self, path: str):
        return LayeredFlowLatentsSampler(
            input_size=self.input_size,
            batch_size=self.batch_size,
            generator=self.generator,
            flow_params=self.params,
            device=self.device,
            latents_mapping=self.mapping,
            mode='eval',
            path=path,
        )
    
    def generate_samples(self, latents, labels):
        images = self.generator(latents, labels=labels).clamp(-1, 1)
        metric_features = [
            metric.get_features(images, labels) for metric in self.metrics
        ]
        optimized_filenames = self.save_images(
            self.experiment_dir,
            images=images,
            labels=labels,
        )
        return _ImageClassifierAttackerOptimizedOutput(
            latents=latents,
            labels=labels,
            metric_features=metric_features,
            scores=None,
            filenames=optimized_filenames,
        )

    def attack(self, targets: list[int]):
        root_path = os.path.join(self.experiment_dir, 'minors')
        latents = []
        labels = []
        for label in targets:
            path = os.path.join(
                root_path, str(label), f'{label}_minor_{self.epochs}.pt'
            )
            sampler = self.trained_flow_sampler(path)
            latents.append(sampler(label, self.eval_bs)[label])
            labels.append(label * torch.ones(self.eval_bs).long())
        latents = torch.cat(latents)
        labels = torch.cat(labels)
        
        optimized_output: _ImageClassifierAttackerOptimizedOutput = batch_apply(
            self.generate_samples,
            latents,
            labels,
            batch_size=self.batch_size,
            description='Optimized Batch',
        )
        
        self._evaluation(
                optimized_output.metric_features,
                optimized_output.labels,
                'optimized',
                self.experiment_dir,
            )
            

    def save_images(self, root_dir: str, images: Tensor, labels: LongTensor):
        assert len(images) == len(labels)

        root_dir = os.path.join(root_dir, 'images')

        all_savenames = []

        for i in range(len(images)):
            image = images[i].detach()
            label = labels[i].item()
            save_dir = os.path.join(root_dir, f'{label}')
            os.makedirs(save_dir, exist_ok=True)
            random_str = get_random_string(length=6)
            save_name = f'{label}_{random_str}.png'
            all_savenames.append(save_name)
            save_path = os.path.join(save_dir, save_name)
            save_image(image, save_path, normalize=True)

        return all_savenames

    def _evaluation(self, features_list, labels, description, save_dir):

        print_split_line(description)

        result = OrderedDict()
        df = pd.DataFrame()
        for features, metric in zip(features_list, self.metrics):

            try:
                for k, v in metric(features, labels).items():
                    result[k] = v
                    df[str(k)] = [v]
                    print_as_yaml({k: v})
            except Exception as e:
                print_split_line()
                print(f'exception metric: {metric.__class__.__name__}')
                traceback.print_exc()
                print_split_line()

        print_split_line()

        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, f'evaluation.csv'), index=None)

        return result