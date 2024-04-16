import os

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as TF


def preprocess_celeba_fn(crop_center, output_resolution):
    if crop_center:
        crop_size = 108
        return TF.Compose(
            [
                TF.CenterCrop((crop_size, crop_size)),
                TF.Resize((output_resolution, output_resolution), antialias=True),
            ]
        )
    else:
        return TF.Resize((output_resolution, output_resolution))


class CelebA(Dataset):

    def __init__(
        self,
        root_path,
        crop_center=False,
        preprocess_resolution=224,
        transform=None,
    ):

        self.preprocess_transform = preprocess_celeba_fn(
            crop_center, preprocess_resolution
        )

        self.dataset = ImageFolder(root=root_path, transform=self.preprocess_transform)
        self.name = 'CelebA'

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, target = self.dataset[idx]
        if self.transform:
            return self.transform(im), target
        else:
            return im, target
