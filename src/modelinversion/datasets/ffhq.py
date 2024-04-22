import os
from typing import Optional, Callable, Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torchvision.transforms as TF


def preprocess_ffhq_fn(crop_center_size, output_resolution):
    if crop_center_size is not None:
        return TF.Compose(
            [
                TF.CenterCrop((crop_center_size, crop_center_size)),
                TF.Resize((output_resolution, output_resolution), antialias=True),
            ]
        )
    else:
        return TF.Resize((output_resolution, output_resolution))


class FFHQ(ImageFolder):

    def __init__(
        self,
        root_path: str,
        crop_center_size: Optional[int] = 800,
        preprocess_resolution: int = 224,
        output_transform: Callable[..., Any] | None = None,
    ):
        preprocess_transform = preprocess_ffhq_fn(
            crop_center_size, preprocess_resolution
        )
        transform = (
            preprocess_transform
            if output_transform is None
            else TF.Compose([preprocess_transform, output_transform])
        )
        super().__init__(root_path, transform)


class FFHQ64(FFHQ):

    def __init__(
        self, root_path: str, output_transform: Callable[..., Any] | None = None
    ):

        super().__init__(root_path, 88, 64, output_transform)


class FFHQ256(FFHQ):

    def __init__(
        self, root_path: str, output_transform: Callable[..., Any] | None = None
    ):

        super().__init__(root_path, 800, 256, output_transform)
