import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Optional, Iterable, Sequence

import torch
from torch import Tensor, LongTensor
from tqdm import tqdm

from ..models import BaseImageGenerator, BaseImageClassifier
from ..utils import batch_apply, print_split_line, print_as_yaml
from .base import SimpleLatentsSampler


class LabelOnlySelectLatentsSampler(SimpleLatentsSampler):

    def __init__(
        self,
        input_size: int | Sequence[int],
        batch_size: int,
        generator: BaseImageGenerator,
        classifier: BaseImageClassifier,
        device: torch.device,
        latents_mapping: Optional[Callable] = None,
        image_transform: Optional[Callable[[Tensor], Tensor]] = None,
        max_iters: int = 100000,
    ) -> None:
        super().__init__(input_size, batch_size, latents_mapping)

        self.generator = generator
        self.classifier = classifier
        self.device = device
        self.image_transform = image_transform
        self.max_iters = max_iters

    def __call__(self, labels: list[int], sample_num: int):

        batch_latent_size = self.get_batch_latent_size(self.batch_size)

        res_labels = set(labels)

        results = defaultdict(list)

        for _ in tqdm(range(self.max_iters)):
            batch_latents = torch.randn(batch_latent_size, device=self.device)
            if self.latents_mapping:
                batch_latents = self.latents_mapping(batch_latents)
            batch_images = self.generator(batch_latents)
            if self.image_transform is not None:
                batch_images = self.image_transform(batch_images)
            pred_scores = self.classifier(batch_images)[0]
            pred_labels = torch.argmax(pred_scores, dim=-1).detach().tolist()
            batch_latents = batch_latents.detach().cpu()

            for i, label in enumerate(pred_labels):
                if label in res_labels:
                    results[label].append(batch_latents[i])
                    if len(results[label]) == sample_num:
                        res_labels.remove(label)
            if len(res_labels) == 0:
                break

        unfinish_labels = []
        res_labels = list(res_labels)

        for label in results:
            results[label] = torch.stack(results[label], dim=0)
            if len(results[label]) < sample_num:
                unfinish_labels.append(label)

        print_split_line('label only unfinish labels')
        print_as_yaml({'no sample labels': res_labels})
        print_as_yaml({'insufficient sample labels': unfinish_labels})
        print_split_line()

        return results
