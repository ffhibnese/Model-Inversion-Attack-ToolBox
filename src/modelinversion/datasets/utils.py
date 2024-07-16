import copy
import random
import os
import shutil
from typing import Sequence, Callable, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
import torch.utils
from torch.utils.data import sampler, Subset
import torch.utils.data
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

import torch
from ..models import BaseImageClassifier
from ..scores import ImageClassificationAugmentConfidence, cross_image_augment_scores
from ..utils import walk_imgs, batch_apply, get_random_string


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2**31


class ClassSubset(Subset):

    def __init__(
        self, dataset: DatasetFolder, target_class: int | Sequence[int]
    ) -> None:

        if isinstance(target_class, int):
            target_class = set((target_class,))
        else:
            target_class = set(target_class)

        targets = dataset.targets
        if (
            hasattr(dataset, 'target_transform')
            and dataset.target_transform is not None
        ):
            targets = [dataset.target_transform(target) for target in dataset.targets]

        indices = [i for i, c in enumerate(targets) if c in target_class]
        super().__init__(dataset, indices)

import torchvision

class IndexImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

@torch.no_grad()
def top_k_selection(
    top_k: int,
    src_dataset_path: str,
    dst_dataset_path: str,
    batch_size: int,
    target_model: BaseImageClassifier,
    num_classes: int,
    device: torch.device,
    create_aug_images_fn: Optional[Callable] = None,
    copy_or_move='copy',
):

    # src_paths = walk_imgs(src_dataset_path)
    ds = IndexImageFolder(src_dataset_path, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # create_aug_images_fn
    ]))
    
    labels = list(range(num_classes))

    if len(ds) < top_k:
        raise RuntimeError(
            f'Find top-{top_k} images, but the src dataset only contains {len(ds)} images.'
        )

    def score_calculate(paths: list[str]):
        totensor = ToTensor()
        imgs = [totensor(Image.open(p)) for p in paths]
        imgs = torch.stack(imgs, dim=0)

        return (
            cross_image_augment_scores(target_model, device, create_aug_images_fn, imgs)
            .detach()
            .cpu()
        )

    # scores = batch_apply(
    #     score_calculate, src_paths, batch_size=batch_size, use_tqdm=True
    # )
    scores = []
    src_paths = []
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    for imgs, _, index in tqdm(dataloader):
        imgs = imgs.to(device)
        scores.append(cross_image_augment_scores(target_model, device, create_aug_images_fn, imgs)
            .detach()
            .cpu())
        for idx in index.cpu().tolist():
            src_paths.append(ds.samples[idx][0])
    scores = torch.cat(scores, dim=0)

    transfer_fn = shutil.copy if copy_or_move == 'copy' else shutil.move

    for label in tqdm(labels):
        dst_dir = os.path.join(dst_dataset_path, f'{label}')
        os.makedirs(dst_dir, exist_ok=True)
        _, indices = torch.topk(scores[:, label], k=top_k)
        indices = indices.tolist()
        for i in range(top_k):
            idx = indices[i]
            src_path = src_paths[idx]
            filename = os.path.split(src_path)[1]
            dst_path = os.path.join(dst_dir, filename)
            transfer_fn(src_path, dst_path)


@torch.no_grad()
def generator_generate_datasets(
    dst_dataset_path: str,
    generator,
    num_per_class: int,
    num_classes: int,
    batch_size: int,
    input_shape: int | tuple,
    target_model: BaseImageClassifier,
    device: torch.device,
):
    # def fn()
    labels = torch.arange(0, num_classes, dtype=torch.long).repeat_interleave(
        num_per_class
    )
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    def get_save_path(label):
        save_dir = os.path.join(dst_dataset_path, f'{label}')
        os.makedirs(save_dir, exist_ok=True)
        filename = get_random_string()
        return os.path.join(save_dir, f'{filename}.png')

    def generation(labels):
        shape = (len(labels), *input_shape)
        labels = labels.to(device)
        z = torch.randn(shape, device=device)
        imgs = generator(z, labels=labels)
        pred = target_model(imgs)[0].argmax(dim=-1).cpu()
        imgs = imgs.cpu()
        for i in range(len(labels)):
            label = pred[i].item()
            save_img = imgs[i]
            savepath = get_save_path(label)
            assert save_img.ndim == 3, save_img.shape
            save_image(save_img, savepath, normalize=True)

    batch_apply(generation, labels, batch_size=batch_size, use_tqdm=True)
