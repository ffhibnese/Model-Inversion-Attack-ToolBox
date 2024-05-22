import os
import yaml
import warnings
import shutil
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Tuple, Callable

import torch
from torch import Tensor, LongTensor
from torchvision.utils import save_image
import pandas as pd

from ..models import *
from ..metrics import *
from ..scores import *
from ..sampler import *
from ..utils import batch_apply, print_as_yaml, print_split_line, get_random_string
from .optimize import BaseImageOptimization


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

        # self.optimized_images = []
        self.metric_features = [[] for _ in range(len(config.eval_metrics))]
        self.final_scores = []
        self.optimized_labels = []
        self.optimized_filenames = []
        self.optimized_latents = []

        os.makedirs(config.save_dir, exist_ok=True)
        self.optimized_save_dir = os.path.join(config.save_dir, 'optimized_images')
        self.final_save_dir = os.path.join(config.save_dir, 'final_images')

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

        # if sample_num < select_num:
        #     warnings.warn('sample_num < select_num. set sample_num = select_num')
        #     sample_num = select_num

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

        return self.concat_tensor_labels(results)

    def concat_tensor_labels(self, target_dict: dict):
        tensors = []
        labels = []
        for target, latents in target_dict.items():
            tensors.append(latents)
            labels += [target] * len(latents)

        labels = LongTensor(labels)
        tensors = torch.cat(tensors, dim=0)
        return tensors, labels

    def concat_optimized_results(self):
        # optimized_images = torch.cat(self.optimized_images, dim=0)
        optimized_metric_features = [torch.cat(f, dim=0) for f in self.metric_features]
        optimized_labels = torch.cat(self.optimized_labels, dim=0)
        final_scores = (
            None if len(self.final_scores) == 0 else torch.cat(self.final_scores)
        )
        optimized_latents = (
            None
            if len(self.optimized_latents) == 0
            else torch.cat(self.optimized_latents, dim=0)
        )
        return (
            optimized_metric_features,
            optimized_labels,
            optimized_latents,
            final_scores,
        )

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

    def update_optimized_images(
        self, images: Tensor, labels: LongTensor, latents: Tensor
    ):

        config = self.config
        assert len(images) == len(labels)
        # self.optimized_images.append(images)
        for dst, metric in zip(self.metric_features, config.eval_metrics):
            dst.append(metric.get_features(images, labels))

        if self.config.final_images_score_fn is not None:
            scores = batch_apply(
                config.final_images_score_fn,
                images,
                labels,
                batch_size=config.final_select_batch_size,
                use_tqdm=True,
            )
            self.final_scores.append(scores)

        self.optimized_labels.append(labels)
        if latents is not None:
            self.optimized_latents.append(latents)

        if self.config.save_optimized_images or self.config.save_final_images:
            self.optimized_filenames += self.save_images(
                self.optimized_save_dir,
                images=images,
                labels=labels,
            )

    def batch_optimize(self, init_latents: Tensor, labels: Tensor):
        images, labels, latents = self.config.optimize_fn(
            init_latents, labels
        ).to_tuple()
        self.update_optimized_images(
            images.detach().cpu(), labels.detach().cpu(), latents.detach().cpu()
        )

    def _evaluation(self, features_list, labels, description):

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

        df.to_csv(os.path.join(self.config.save_dir, f'{description}.csv'), index=None)

        return result

    def save_images(self, root_dir: str, images: Tensor, labels: LongTensor):
        assert len(images) == len(labels)

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

    def save_selection_images(self, indices_dict: dict[int, list[str]]):
        for target, indices in indices_dict.items():
            src_dir = os.path.join(self.optimized_save_dir, f'{target}')
            dst_dir = os.path.join(self.final_save_dir, f'{target}')
            os.makedirs(dst_dir, exist_ok=True)
            for idx in indices:
                filename = self.optimized_filenames[idx]
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)

    def save_final_selection_latents(
        self, indices_dict: dict[int, list[str]], all_latents
    ):
        if all_latents is None or len(all_latents) == 0:
            return
        labels = []
        indices = []
        for target, target_indices in indices_dict.items():
            labels += [target] * len(target_indices)
            indices += target_indices

        labels = np.array(labels, dtype=np.int32)
        indices = np.array(indices, dtype=np.int32)
        if isinstance(all_latents, Tensor):
            all_latents = all_latents.numpy()
        latents = all_latents[indices]
        np.save(os.path.join(self.config.save_dir, 'final_latents.npy'), latents)
        np.save(os.path.join(self.config.save_dir, 'final_labels.npy'), labels)

    def get_final_selection_features_labels(
        self, indices_dict: dict[int, list[str]], features_list
    ):

        res_indices = []
        res_labels = []

        for target, indices in indices_dict.items():
            indices = torch.LongTensor(indices)
            labels = torch.ones_like(indices) * target

            res_indices.append(indices)
            res_labels.append(labels)

        res_indices = torch.cat(res_indices, dim=0)
        res_labels = torch.cat(res_labels, dim=0)

        return [feature[res_indices] for feature in features_list], res_labels

    def attack(self, target_list: list[int]):
        config = self.config
        os.makedirs(config.save_dir, exist_ok=True)

        # print_split_line('Attack Config')
        # print(config)
        # print_split_line()

        # initial selection for each target
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
        batch_apply(
            self.batch_optimize,
            init_latents,
            init_labels,
            batch_size=config.optimize_batch_size,
            description='Optimized Batch',
        )

        # concat optimized images and labels
        (
            optimized_metric_features,
            optimized_labels,
            optimized_latents,
            optimized_scores,
        ) = self.concat_optimized_results()

        if self.config.save_optimized_images and optimized_latents is not None:
            np.save(
                os.path.join(self.config.save_dir, f'optimized_latents.npy'),
                optimized_latents.numpy(),
            )
            np.save(
                os.path.join(self.config.save_dir, f'optimized_labels.npy'),
                optimized_labels.numpy(),
            )

        if self.config.eval_optimized_result or (
            optimized_scores is None and config.eval_final_result
        ):
            print('evaluate optimized result')
            self._evaluation(
                optimized_metric_features,
                optimized_labels,
                'Optimized-Image-Evaluation',
            )

        # # final selection
        if optimized_scores is not None:
            print('execute final selection')
            final_res = self.final_selection(
                config.final_num,
                optimized_scores,
                optimized_labels,
            )

            if config.save_final_images:
                print('save final images')

                self.save_selection_images(final_res)
                self.save_final_selection_latents(final_res, optimized_latents)

            if self.config.eval_final_result:
                print('evaluate final result')
                final_features, final_labels = self.get_final_selection_features_labels(
                    final_res, optimized_metric_features
                )
                self._evaluation(final_features, final_labels, 'Final-Image-Evaluation')


# class PostMetricCalculator:

#     def __init__(
#         self,
#         experiment_dir: str,
#         metrics: list[BaseImageMetric],
#         generator: BaseImageGenerator,
#     ) -> None:
#         self.experiment_dir = experiment_dir
#         self.datas = []
#         self.metrics = metrics

from collections import defaultdict


def post_metric_calculate(
    experiment_dir: str,
    batch_size: int,
    metrics: list[BaseImageMetric],
    generator: BaseImageGenerator,
    device: torch.device,
):
    datas = defaultdict(dict)
    filenames = [
        fname
        for fname in os.listdir(experiment_dir)
        if fname.endswith('.npy') and '_' in fname
    ]
    for fname in filenames:
        result_description, data_type = fname.rsplit('_', 1)
        if data_type == 'labels.npy':
            datas[result_description]['labels'] = os.path.join(experiment_dir, fname)
        elif data_type == 'latents.npy':
            datas[result_description]['latents'] = os.path.join(experiment_dir, fname)

    for result_description, info_fnames in datas.items():
        if len(info_fnames) != 2:
            continue

        latents = torch.from_numpy(np.load(info_fnames['latents']))
        labels = torch.from_numpy(np.load(info_fnames['labels']))

        def _generate(latents, labels):
            return generator(latents.to(device), labels=labels.to(device)).cpu()

        print_split_line(result_description)
        images = batch_apply(
            _generate, latents, labels, batch_size=batch_size, use_tqdm=True
        )

        result_dict = OrderedDict()
        for metric in metrics:
            features = metric.get_features(images, labels)
            metric_ret = metric(features, labels)
            for k, v in metric_ret.items():
                result_dict[k] = v

        print_as_yaml(result_dict)
