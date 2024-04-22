import copy
import random
import os
import shutil
from typing import Sequence, Callable, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import sampler, Subset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

import torch
from ..models import BaseImageClassifier
from ..scores import ImageClassificationAugmentConfidence, cross_image_augment_scores
from ..utils import walk_imgs, batch_apply, get_random_string

# class RandomIdentitySampler(sampler.Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#     """

#     def __init__(self, dataset, batch_size, num_instances=1):
#         self.data_source = dataset
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.num_pids_per_batch = self.batch_size // self.num_instances
#         self.index_dic = defaultdict(list)
#         # changed according to the dataset
#         for index, inputs in enumerate(self.data_source):
#             self.index_dic[inputs[1]].append(index)

#         self.pids = list(self.index_dic.keys())

#         # estimate number of examples in an epoch
#         self.length = 0
#         for pid in self.pids:
#             idxs = self.index_dic[pid]
#             num = len(idxs)
#             if num < self.num_instances:
#                 num = self.num_instances
#             self.length += num - num % self.num_instances

#     def __iter__(self):
#         batch_idxs_dict = defaultdict(list)

#         for pid in self.pids:
#             idxs = copy.deepcopy(self.index_dic[pid])
#             if len(idxs) < self.num_instances:
#                 idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
#             random.shuffle(idxs)
#             batch_idxs = []
#             for idx in idxs:
#                 batch_idxs.append(idx)
#                 if len(batch_idxs) == self.num_instances:
#                     batch_idxs_dict[pid].append(batch_idxs)
#                     batch_idxs = []

#         avai_pids = copy.deepcopy(self.pids)
#         final_idxs = []

#         while len(avai_pids) >= self.num_pids_per_batch:
#             selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
#             for pid in selected_pids:
#                 batch_idxs = batch_idxs_dict[pid].pop(0)
#                 final_idxs.extend(batch_idxs)
#                 if len(batch_idxs_dict[pid]) == 0:
#                     avai_pids.remove(pid)

#         self.length = len(final_idxs)
#         return iter(final_idxs)

#     def __len__(self):
#         return self.length


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

    src_paths = walk_imgs(src_dataset_path)
    labels = list(range(num_classes))

    if len(src_paths) < top_k:
        raise RuntimeError(
            f'Find top-{top_k} images, but the src dataset only contains {len(src_paths)} images.'
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

    scores = batch_apply(
        score_calculate, src_paths, batch_size=batch_size, use_tqdm=True
    )

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
        img = generator(z, labels=labels)
        pred = target_model(img)[0].argmax(dim=-1).cpu()
        img = img.cpu()
        for i in range(len(labels)):
            label = pred[i].item()
            save_im = img[i]
            savepath = get_save_path(label)
            assert save_im.ndim == 3, save_im.shape
            save_image(img, savepath, normalize=True)

    batch_apply(generation, labels, batch_size=batch_size, use_tqdm=True)
