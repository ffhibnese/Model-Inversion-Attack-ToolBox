import os
import yaml
import warnings
import shutil
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Tuple, Callable, Dict

import torch
from torch import Tensor, LongTensor
from torchvision.utils import save_image
import pandas as pd

from ..models import *
from ..metrics import *
from ..scores import *
from ..sampler import *
from ..utils import (
    batch_apply,
    print_as_yaml,
    print_split_line,
    get_random_string,
    BaseOutput,
    safe_save
)
from .optimize import BaseImageOptimization


def label_dict_to_pairs(label_dict: Dict[int, Tensor | list[int]]):
    tensors = []
    labels = []
    for target, latents in label_dict.items():
        if not isinstance(latents, Tensor):
            if isinstance(latents[0], int):
                latents = LongTensor(latents)
            else:
                latents = Tensor(latents)
        tensors.append(latents)
        labels += [target] * len(latents)

    labels = LongTensor(labels)
    tensors = torch.cat(tensors, dim=0)
    return tensors, labels


@dataclass
class ImageClassifierAttackConfig:
    """This is the configuration class to store the configuration of a [`ImageClassifierAttacker`]. It is used to instantiate an Attacker according to the specified arguments.

    Args:
        latents_sampler (BaseLatentsSampler):
            The sampler for generating latent vectors for each label.
        initial_num (int, optional):
            The number of latent vectors for each label at the time of initial selection. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        initial_latents_score_fn (BaseLatentScore, optional):
            The function to generate scores of each latents vectors for each class. The attacker will select `optimize_num` latent vectors with the highest scores for each class in the initial selection process. No initial selection when it is `None`. Defaults to `None`.
        initial_select_batch_size (int, optional):
            Batch size in the initial process. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        optimize_num (int):
            The number of latent vectors for each label at the time of optimization. Defaults to 50.
        optimize_fn (BaseImageOptimization):
            The function to generate images and final labels from initial latent vectors and labels.
        optimize_batch_size (int):
            Batch size in the optimization process. It will keep the same as `optimize_num` when it is `None`. Defaults to 5.
        final_num (int, optional):
            The number of latent vectors for each label at the time of final selection. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        final_images_score_fn (BaseImageClassificationScore, optional):
            The function to generate scores of images for the given labels. The attacker will select `final_num` images with the highest scores for the given label for each class in the final selection process. No final selection when it is `None`. Defaults to `None`.
        final_select_batch_size (int, optional):
            Batch size in the final process. It will keep the same as `optimize_num` when it is `None`. Defaults to `None`.
        save_dir (str, optional):
            The file folder to save inversed images. Defaults to `None`.
        save_optimized_images (bool, optional):
            Whether to save the optimized images. Defaults to False.
        save_final_images (bool, optional):
            Whether to save the final images. Defaults to False.
        save_kwargs (dict, optional):
            Other args for saving images. Defaults to `{}`.
        eval_metrics (list[BaseImageMetric], optional):
            A list of metric methods. Defaults to `[]`.
        eval_optimized_result (bool, optional):
            Whether to evaluate the optimized results. Defaults to False.
        eval_final_result (bool, optional):
            Whether to evaluate the final results. Defaults to False.
    """

    # sample latent
    latents_sampler: BaseLatentsSampler

    # # initial selection
    initial_num: Optional[int] = None
    initial_latents_score_fn: Optional[BaseLatentScore] = None
    initial_select_batch_size: Optional[int] = None

    # optimzation & generate images
    optimize_num: int = 50
    optimize_fn: BaseImageOptimization = None
    optimize_batch_size: int = 5

    # final selection
    final_num: Optional[int] = None
    final_images_score_fn: Optional[BaseImageClassificationScore] = None
    final_select_batch_size: Optional[int] = None

    # save
    save_dir: Optional[str] = None
    save_optimized_images: bool = False
    save_final_images: bool = False
    save_kwargs: dict = field(default_factory=lambda: {})

    # metric
    eval_metrics: list[BaseImageMetric] = field(default=lambda: [])
    eval_optimized_result: bool = False
    eval_final_result: bool = False


@dataclass
class _ImageClassifierAttackerOptimizedOutput(BaseOutput):
    latents: Tensor
    labels: LongTensor
    metric_features: list[Tensor]
    scores: Tensor
    filenames: list[str]


class ImageClassifierAttacker(ABC):
    """This is the model inversion attack class for image classification task.

    Args:
        config (ImageClassifierAttackConfig): ImageClassifierAttackConfig.


    Attack Flow

                         Start
                           |
                           |
        +------------------v------------------+
        |                                     |
        |           Latent Sampler            |
        |       Inital Latents Generation     |
        |                                     |
        +-------+----------+----------+-------+                                                                          *
                |          |          |
                |          |          |
        +-------v----------v----------v-------+
        |                                     |
        |           Initial Selection         |
        |       Select Latents By Scores      |
        |                                     |
        +-------+----------+----------+-------+
                |          |          |
                |          |          |
        +-------v----------v----------v-------+
        |                                     |
        |             Optimization            |
        |     O ptimize Latents To Images     |
        |                                     |
        +-------+----------+----------+-------+
                |          |          |
                |          |          |
        +-------v----------v----------v-------+
        |                                     |
        |           Final Selection           |
        |       Select Images By Scores       |
        |                                     |
        +------------------+------------------+
                           |
                           |
                           v
                          End
    """

    def __init__(self, config: ImageClassifierAttackConfig) -> None:
        self.config = self._preprocess_config(config)

        os.makedirs(config.save_dir, exist_ok=True)

        self.optimized_save_dir = os.path.join(config.save_dir, 'optimized')
        self.final_save_dir = os.path.join(config.save_dir, 'final')

    def _preprocess_config(self, config: ImageClassifierAttackConfig):

        if (
            config.save_optimized_images or config.save_final_images
        ) and not config.save_dir:
            raise RuntimeError('`save_dir` is not set')

        if config.latents_sampler is None:
            raise RuntimeError('`latents_sampler` cannot be None')

        if config.optimize_fn is None:
            raise RuntimeError('`optimize_fn` cannot be None')

        if config.optimize_num is None:
            raise RuntimeError('`optimize_num` cannot be None')

        if config.initial_num is None:
            config.initial_num = config.optimize_num

        if config.final_num is None:
            config.final_num = config.optimize_num

        # if config.final_num > config.optimize_num:
        #     warnings.warn('the final number is larger than the optimize number, automatically set the latter to the fronter')
        #     config.optimize_num = config.final_num

        if config.optimize_num > config.initial_num:
            warnings.warn(
                'the optimize number is larger than the initial number, automatically set the latter to the fronter'
            )
            config.initial_num = config.optimize_num

        if config.initial_select_batch_size is None:
            config.initial_select_batch_size = config.optimize_batch_size

        if config.final_select_batch_size is None:
            config.final_select_batch_size = config.final_select_batch_size

        return config

    # def post_evaluation(self, metric)

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
            save_image(image, save_path, **self.config.save_kwargs)

        return all_savenames

    def initial_latents(
        self,
        batch_size: int,
        sample_num: int,
        select_num: int,
        labels: list[int],
        latent_score_fn: Optional[Callable] = None,
    ) -> dict[int, Tensor]:

        if isinstance(labels, Tensor):
            labels = labels.tolist()
        latents_dict = self.config.latents_sampler(labels, sample_num)

        if latent_score_fn is None or sample_num == select_num:
            if sample_num > select_num:
                warnings.warn(
                    'no score function, automatically sample `select_num` latents'
                )
            results = latents_dict
        else:
            results = {}
            for label in tqdm(labels):
                latents = latents[label]
                labels = torch.ones((len(latents),), dtype=torch.long)
                # scores = latent_score_fn(latents, labels)
                scores = batch_apply(
                    latent_score_fn, latents, labels, batch_size=batch_size
                )
                _, indices = torch.topk(scores, k=select_num)
                results[label] = latents[indices]

        return label_dict_to_pairs(results)

    def _batch_optimize(self, latents: Tensor, labels: LongTensor):
        config = self.config

        images, labels, latents = self.config.optimize_fn(latents, labels).to_tuple()

        metric_features = [
            metric.get_features(images, labels) for metric in self.config.eval_metrics
        ]

        scores = None
        if self.config.final_images_score_fn is not None:
            scores = batch_apply(
                config.final_images_score_fn,
                images,
                labels,
                batch_size=config.final_select_batch_size,
                use_tqdm=True,
            )

        optimized_filenames = None
        if self.config.save_optimized_images or self.config.save_final_images:
            optimized_filenames = self.save_images(
                self.optimized_save_dir,
                images=images,
                labels=labels,
            )

        return _ImageClassifierAttackerOptimizedOutput(
            latents=latents,
            labels=labels,
            metric_features=metric_features,
            scores=scores,
            filenames=optimized_filenames,
        )

    def optimize(self, latents: Tensor, labels: LongTensor, batch_size: int):
        return batch_apply(
            self._batch_optimize,
            latents,
            labels,
            batch_size=batch_size,
            description='Optimized Batch',
        )

    def _evaluation(self, features_list, labels, description, save_dir):

        print_split_line(description)

        result = OrderedDict()
        df = pd.DataFrame()
        for features, metric in zip(features_list, self.config.eval_metrics):

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

    def final_selection(
        self,
        final_num: int,
        scores: Tensor,
        labels: LongTensor,
    ):

        assert len(scores) == len(labels)

        targets = set(labels.tolist())

        print('final selection: select by scores')

        results = {}

        for target in targets:
            indices = torch.nonzero(labels == target, as_tuple=True)[0]
            indices_list = indices.detach().cpu().numpy().reshape(-1).tolist()
            target_scores = scores[indices]
            if len(target_scores) <= final_num:
                results[target] = indices_list
            else:
                _, topk_idx = torch.topk(target_scores, k=final_num)
                topk_idx = topk_idx.detach().cpu().numpy().reshape(-1).tolist()
                indices_list = [indices_list[i] for i in topk_idx]
                results[target] = indices_list
            # results[target] = target_images[topk_idx]

        return results

    def attack(self, target_list: list[int]):

        config = self.config

        print('execute initial selection')
        init_latents, init_labels = self.initial_latents(
            config.initial_select_batch_size,
            config.initial_num,
            config.optimize_num,
            target_list,
            config.initial_latents_score_fn,
        )

        # execute optimize
        print('execute optimization')
        optimized_output: _ImageClassifierAttackerOptimizedOutput = self.optimize(
            latents=init_latents,
            labels=init_labels,
            batch_size=config.optimize_batch_size,
        )

        if config.save_optimized_images and optimized_output.latents is not None:
            save_dir = os.path.join(self.optimized_save_dir, 'cache')
            os.makedirs(save_dir, exist_ok=True)

            np.save(
                os.path.join(save_dir, f'latents.npy'),
                optimized_output.latents.numpy(),
            )
            np.save(
                os.path.join(save_dir, f'labels.npy'),
                optimized_output.labels.numpy(),
            )

        if config.eval_optimized_result or (
            optimized_output.scores is None and config.eval_final_result
        ):
            print('evaluate optimized result')
            self._evaluation(
                optimized_output.metric_features,
                optimized_output.labels,
                'optimized',
                self.optimized_save_dir,
            )

        if optimized_output.scores is not None:
            print('execute final selection')
            final_label_indices_dict = self.final_selection(
                config.final_num,
                optimized_output.scores,
                optimized_output.labels,
            )

            if config.save_final_images:
                print('save final images')

                self.save_selection_images(
                    final_label_indices_dict, optimized_output.filenames
                )

            final_indices, final_labels = label_dict_to_pairs(final_label_indices_dict)

            if optimized_output.latents is not None:
                assert final_indices.ndim == 1
                final_latents = optimized_output.latents[final_indices]

                save_dir = os.path.join(self.final_save_dir, 'cache')
                os.makedirs(save_dir, exist_ok=True)

                np.save(
                    os.path.join(save_dir, f'latents.npy'),
                    final_latents.numpy(),
                )
                np.save(
                    os.path.join(save_dir, f'labels.npy'),
                    final_labels.numpy(),
                )

            if config.eval_final_result:
                print('evaluate final result')
                final_features = [
                    features[final_indices]
                    for features in optimized_output.metric_features
                ]
                self._evaluation(
                    final_features, final_labels, 'final', self.final_save_dir
                )

    def save_selection_images(
        self, indices_dict: dict[int, list[str]], filenames: list[str]
    ):
        for target, indices in indices_dict.items():
            src_dir = os.path.join(self.optimized_save_dir, 'images', f'{target}')
            dst_dir = os.path.join(self.final_save_dir, 'images', f'{target}')
            os.makedirs(dst_dir, exist_ok=True)
            for idx in indices:
                filename = filenames[idx]
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)

    def evaluate_from_pre_generate(
        self,
        generator: BaseImageGenerator,
        select_classes: Optional[list[int]] = None,
        description: str = 'alteval',
        device: torch.device = 'cpu',
    ):
        config = self.config
        for foldername in os.listdir(config.save_dir):
            folder = os.path.join(config.save_dir, foldername)
            cache_folder = os.path.join(folder, 'cache')
            if not os.path.isdir(cache_folder):
                continue
            labels_file = os.path.join(cache_folder, 'labels.npy')
            latents_file = os.path.join(cache_folder, 'latents.npy')

            if not os.path.exists(labels_file) or not os.path.exists(latents_file):
                continue

            print_split_line(foldername)

            labels = torch.from_numpy(np.load(labels_file)).long()
            latents = torch.from_numpy(np.load(latents_file))

            if select_classes is not None:
                select_indices = torch.zeros_like(labels, dtype=torch.bool)
                for target in select_classes:
                    select_indices |= labels == target
                labels = labels[select_indices]
                latents = latents[select_indices]

            print(len(labels))

            def _batch_get_features(latents, labels):
                images = generator(latents.to(device), labels=labels.to(device)).cpu()
                return [
                    metric.get_features(images, labels)
                    for metric in config.eval_metrics
                ]

            metric_features = batch_apply(
                _batch_get_features,
                latents,
                labels,
                batch_size=config.optimize_batch_size,
            )

            self._evaluation(
                metric_features,
                labels,
                description=f'{foldername}_{description}',
                save_dir=f'{folder}_{description}',
            )
