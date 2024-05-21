import os
import warnings
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Optional, Callable

import torch
import numpy as np
from numpy import ndarray
from scipy import linalg
from torch import Tensor, nn, LongTensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
import pandas as pd
import traceback

# from ..foldermanager import FolderManager
from ..models import BaseImageClassifier, BaseImageEncoder, HOOK_NAME_FEATURE
from ..datasets.utils import ClassSubset
from ..utils import (
    batch_apply,
    safe_save_csv,
    unwrapped_parallel_module,
    print_split_line,
)
from .fid import fid_utils


class BaseImageMetric(ABC):
    """Base class for all image metric classes.

    Args:
        batch_size (int): Batch size when executing the metric.
    """

    def __init__(self, batch_size: int, transform: Optional[Callable] = None):
        self.batch_size = batch_size
        self.transform = transform

    @abstractmethod
    def _get_features_impl(self, images: Tensor, labels: LongTensor) -> Tensor:
        pass

    def get_features(self, images: Tensor, labels: LongTensor) -> Tensor:

        def _batch_get_features(images: Tensor, labels: LongTensor):
            if self.transform is not None:
                images = self.transform(images)
            return self._get_features_impl(images, labels)

        return batch_apply(
            _batch_get_features, images, labels, batch_size=self.batch_size
        )

    @abstractmethod
    def _call_impl(self, features: Tensor, labels: LongTensor) -> OrderedDict:
        pass

    def __call__(self, features: Tensor, labels: LongTensor) -> OrderedDict:
        """Executing the evaluation for inversed images with the given labels.

        Args:
            features (Tensor): Features of inversed images
            labels (LongTensor): Labels for the corresponding features.

        Returns:
            OrderedDict: Results of the metric.
        """
        return self._call_impl(features, labels)


class ImageClassifierAttackAccuracy(BaseImageMetric):
    """Attack accuracy metric for inversed images.

    Args:
        batch_size (int): Batch size when executing the metric.
        model (BaseImageClassifier): The evaluation image classifier.
        device (device): Device to run the metric. It should be kept the same with the device of the model.
        description (str): Prefix of the metric.
    """

    def __init__(
        self,
        batch_size: int,
        model: BaseImageClassifier,
        device: torch.device,
        description: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(batch_size, transform)
        self.model = model
        self.device = device
        self.description = description

    @torch.no_grad()
    def _get_features_impl(self, images: Tensor, labels: LongTensor) -> Tensor:

        def get_scores(images: Tensor):
            images = images.to(self.device)
            pred, _ = self.model(images)
            return pred.cpu().detach()

        scores = batch_apply(
            get_scores, images, batch_size=self.batch_size, use_tqdm=True
        )

        _, topk_indices = torch.topk(scores, 5)
        eq = (topk_indices == labels.unsqueeze(1)).float()
        acc = eq[:, 0]
        acc5 = eq.sum(dim=-1)

        return torch.stack([acc, acc5], dim=-1)

    @torch.no_grad()
    def _call_impl(self, features: Tensor, labels: LongTensor) -> OrderedDict:

        accs, acc5s = features[:, 0].reshape(-1), features[:, 1].reshape(-1)

        acc, acc5 = accs.mean().item(), acc5s.mean().item()

        ret = OrderedDict(
            [(f'{self.description} acc@1', acc), (f'{self.description} acc@5', acc5)]
        )

        try:
            target_values = list(set(labels.cpu().tolist()))

            target_accs = []
            target_acc5s = []
            target_numbers = []
            max_nums = 0

            for step, target in enumerate(tqdm(target_values, leave=False)):
                target_idx = labels == target
                target_acc = accs[target_idx]  # .mean().item()
                target_acc5 = acc5s[target_idx]  # .mean().item()
                target_accs.append(target_acc)
                target_acc5s.append(target_acc5)
                target_numbers.append(len(target_acc))
                max_nums = max(max_nums, len(target_acc))

            weights = torch.zeros(
                (max_nums,), dtype=features.dtype, device=features.device
            )
            acc_cumsum = torch.zeros(
                (max_nums,), dtype=features.dtype, device=features.device
            )
            acc5_cumsum = torch.zeros(
                (max_nums,), dtype=features.dtype, device=features.device
            )
            mask_ranges = torch.arange(0, max_nums, device=features.device)
            for target_acc, target_acc5, target_num in zip(
                target_accs, target_acc5s, target_numbers
            ):
                if target_num == 0:
                    continue
                mask = (mask_ranges < target_num).to(weights.dtype)
                weights += mask
                # print('>>', target_num, target_acc.shape)
                acc_cumsum[:target_num] += target_acc
                acc5_cumsum[:target_num] += target_acc5

            div_weights = torch.clamp_min(weights, torch.ones_like(weights))
            acc_mean = (acc_cumsum / div_weights).cpu().numpy()
            acc5_mean = (acc5_cumsum / div_weights).cpu().numpy()

            acc_std = np.std(acc_mean, axis=0).mean()
            acc5_std = np.std(acc5_mean, axis=0).mean()
            ret[f'{self.description} acc@1 std'] = float(acc_std)
            ret[f'{self.description} acc@5 std'] = float(acc5_std)
        except Exception as e:
            print_split_line()
            traceback.print_exc()
            print_split_line()

        return ret


class ImageDistanceMetric(BaseImageMetric):
    """Distance metrics for each target class.

    Args:
        batch_size (int): Batch size when executing the metric.
        model (BaseImageClassifier): The evaluation image classifier.
        dataset (DatasetFolder): The private dataset.
        device (device): Device to run the metric. It should be kept the same with the device of the model.
        description (str): Prefix of the metric.
        save_individual_res_dir (str, optional): File folder to save results for each class if it existed. Default to `None`.
        num_workers (int), `num_workers` of the data loader. Default to 8.
    """

    def __init__(
        self,
        batch_size: int,
        model: BaseImageClassifier,
        dataset: DatasetFolder,
        device: torch.device,
        description: str,
        transform: Optional[Callable] = None,
        save_individual_res_dir: Optional[str] = None,
        num_workers=8,
    ):
        super().__init__(batch_size, transform)

        self.model = model
        self.dataset = dataset
        self.device = device
        self.description = description
        self.save_dir = save_individual_res_dir
        # self.hook = unwrapped_parallel_module(model).get_last_feature_hook()
        self.num_workers = num_workers

    @torch.no_grad()
    def _get_features_impl(self, images: Tensor, labels: LongTensor):
        images = images.to(self.device)
        # self.hook.clear_feature()
        _, hook_res = self.model(images)
        if HOOK_NAME_FEATURE not in hook_res:
            raise RuntimeError(
                f'The model has not registered the hook for {HOOK_NAME_FEATURE}'
            )
        feature = hook_res[HOOK_NAME_FEATURE]
        # print(images.shape, feature.shape)
        return feature.reshape(len(images), -1).detach().cpu()

    @torch.no_grad()
    def _call_impl(self, features: Tensor, labels: LongTensor) -> OrderedDict:

        target_values = list(set(labels.cpu().tolist()))

        target_dists = []
        target_nums = []

        for step, target in enumerate(tqdm(target_values, leave=False)):
            target_src_features = features[labels == target]

            target_dst_ds = ClassSubset(self.dataset, target)
            target_dst_features = []
            for dst_img, _ in DataLoader(
                target_dst_ds,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ):
                target_dst_features.append(
                    self._get_features_impl(dst_img, None).detach()
                )
            target_dst_features = torch.cat(target_dst_features, dim=0)

            distance = torch.cdist(target_src_features, target_dst_features) ** 2
            distance, _ = torch.min(distance, dim=1)

            target_dists.append(distance.mean().item())
            target_nums.append(len(distance))

        target_values = np.array(target_values, dtype=np.int32)
        target_dists = np.array(target_dists)
        target_nums = np.array(target_nums)

        if self.save_dir is not None:
            df = pd.DataFrame()
            df['target'] = target_values
            df['square distance'] = target_dists
            save_name = f'{self.description}_square_distance.csv'
            safe_save_csv(df, self.save_dir, save_name)

        result = (target_dists * target_nums).sum() / target_nums.sum()
        ret = OrderedDict([[f'{self.description} square distance', float(result)]])
        try:
            target_dists_std = np.std(target_dists, axis=0).mean()
            ret[f'{self.description} square distance std'] = float(target_dists_std)
        except:

            print_split_line()
            traceback.print_exc()
            print_split_line()
        return ret


class ImageFidPRDCMetric(BaseImageMetric):
    """A class for calculating FID and PRDC (Precision, Recall, Diversity and Coverage). The model will use InceptionV3 pretrained with ImageNet.

    Args:
        batch_size (int): Batch size when executing the metric.
        dataset (DatasetFolder): The private dataset.
        device (device): Device to run the metric. It should be kept the same with the device of the model.
        description (str): Prefix of the metric.
        save_individual_prdc_dir (str, optional): File folder to save PRDC results for each class if it existed. Default to `None`.
        num_workers (int), `num_workers` of the data loader. Default to 8.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: DatasetFolder,
        device: torch.device,
        transform: Optional[Callable] = None,
        prdc_k=3,
        fid=True,
        prdc=True,
        save_individual_prdc_dir: Optional[str] = None,
        num_workers=8,
        description='incv3',
    ):
        super().__init__(batch_size, transform)

        self.device = device
        self.dataset = dataset
        self.inception_model = InceptionV3([InceptionV3.DEFAULT_BLOCK_INDEX]).to(
            self.device
        )
        self.num_workers = num_workers
        self.prdc_k = prdc_k
        self.description = description

        self.calc_fid = fid
        self.calc_prdc = prdc
        self.save_dir = save_individual_prdc_dir

    @torch.no_grad()
    def _calculate_activation_statistics(self, dataset, use_tqdm=False):
        dataloader = DataLoader(
            dataset, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        pred_arr = []
        labels_arr = []

        if use_tqdm:
            dataloader = tqdm(dataloader, leave=False)
        for image, labels in dataloader:
            labels_arr.append(labels)
            image = image.to(self.device)
            pred = self.inception_model(image)[0].squeeze(3).squeeze(2).detach().cpu()
            pred_arr.append(pred)
        pred_arr = torch.cat(pred_arr, dim=0)
        labels_arr = torch.cat(labels_arr, dim=0)
        # pred_numpy = pred_arr.numpy()
        # labels_numpy = labels_arr.numpy()

        return (
            pred_arr,
            labels_arr,
            # np.mean(pred_numpy, axis=0),
            # np.cov(pred_numpy, rowvar=False),
        )

    def _get_features_impl(self, images: Tensor, labels: LongTensor) -> Tensor:
        src_ds = TensorDataset(images, labels)
        return self._calculate_activation_statistics(src_ds, use_tqdm=False)[0]

    @torch.no_grad()
    def _call_impl(self, features: Tensor, labels: LongTensor) -> OrderedDict:

        target_values = list(set(labels.cpu().tolist()))
        # src_ds = TensorDataset(images, labels)
        dst_ds = ClassSubset(self.dataset, target_values)

        # fake_feature, fake_labels, mu_fake, sigma_fake = (
        #     self._calculate_activation_statistics(src_ds)
        # )
        fake_feature = features
        fake_labels = labels

        fake_feature_np = fake_feature.detach().cpu().numpy()
        mu_fake, sigma_fake = np.mean(fake_feature_np, axis=0), np.cov(
            fake_feature_np, rowvar=False
        )

        real_feature, real_labels = self._calculate_activation_statistics(
            dst_ds, use_tqdm=True
        )

        real_feature_np = real_feature.numpy()
        mu_real, sigma_real = np.mean(real_feature_np, axis=0), np.cov(
            real_feature_np, rowvar=False
        )

        # print(
        #     f'fake shapes: {fake_feature_np.shape} {mu_fake.shape} {sigma_fake.shape}'
        # )
        # print(
        #     f'real shapes: {real_feature_np.shape} {mu_real.shape} {sigma_real.shape}'
        # )

        result = OrderedDict()

        # FID
        if self.calc_fid:
            fid_score = fid_utils.calculate_frechet_distance(
                mu_fake, sigma_fake, mu_real, sigma_real
            )
            result['FID'] = float(fid_score)

        # PRDC
        if self.calc_prdc:
            target_list = []
            precision_list = []
            recall_list = []
            density_list = []
            coverage_list = []

            unfinish_list = []
            for target in tqdm(target_values, leave=False):
                fake_mask = fake_labels == target
                real_mask = real_labels == target
                embedding_fake = fake_feature[fake_mask]
                embedding_real = real_feature[real_mask]

                if (
                    len(embedding_fake) <= self.prdc_k + 1
                    or len(embedding_real) <= self.prdc_k + 1
                ):
                    unfinish_list.append(target)
                    continue
                target_list.append(target)

                pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
                pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
                pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
                pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
                radius_real = pair_dist_real[:, self.prdc_k]
                radius_fake = pair_dist_fake[:, self.prdc_k]

                # Compute precision
                distances_fake_to_real = torch.cdist(
                    embedding_fake, embedding_real, p=2
                )
                min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
                precision = (
                    (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
                )
                precision_list.append(precision.cpu().item())

                # Compute recall
                distances_real_to_fake = torch.cdist(
                    embedding_real, embedding_fake, p=2
                )
                min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
                recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
                recall_list.append(recall.cpu().item())

                # Compute density
                num_samples = distances_fake_to_real.shape[0]
                sphere_counter = (
                    (distances_fake_to_real <= radius_real.repeat(num_samples, 1))
                    .float()
                    .sum(dim=0)
                    .mean()
                )
                density = sphere_counter / self.prdc_k
                density_list.append(density.cpu().item())

                # Compute coverage
                num_neighbors = (
                    (distances_fake_to_real <= radius_real.repeat(num_samples, 1))
                    .float()
                    .sum(dim=0)
                )
                coverage = (num_neighbors > 0).float().mean()
                coverage_list.append(coverage.cpu().item())

            target_values = np.array(target_list, dtype=np.int32)
            precision = np.array(precision_list)
            recall = np.array(recall_list)
            density = np.array(density_list)
            coverage = np.array(coverage_list)

            if len(unfinish_list) != 0:
                warnings.warn(
                    f'The number of images for those classes are too small, skip the evaluation: {unfinish_list}'
                )

            result['precision'] = float(precision.mean())
            result['recall'] = float(recall.mean())
            result['density'] = float(density.mean())
            result['coverage'] = float(coverage.mean())

            try:
                # print('>>>>>>>>>>>>>>>>>> std prdc')
                if self.save_dir is not None:
                    df = pd.DataFrame()
                    df['target'] = target_values
                    df['precision'] = precision
                    df['recall'] = recall
                    df['density'] = density
                    df['coverage'] = coverage
                    save_name = f'{self.description}_prdc.csv'
                    safe_save_csv(df, self.save_dir, save_name)
                result['precision std'] = float(np.std(precision, axis=0).mean())
                result['recall std'] = float(np.std(recall, axis=0).mean())
                result['density std'] = float(np.std(density, axis=0).mean())
                result['coverage std'] = float(np.std(coverage, axis=0).mean())
            except:

                print_split_line()
                traceback.print_exc()
                print_split_line()

        return result
