import os

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as TF


def preprocess_facescrub_fn(crop_center, output_resolution):
    if crop_center:
        crop_size = int(54 * output_resolution / 64)
        return TF.Compose(
            [
                TF.Resize((output_resolution, output_resolution), antialias=True),
                TF.CenterCrop((crop_size, crop_size)),
                TF.Resize((output_resolution, output_resolution), antialias=True),
            ]
        )
    else:
        return TF.Resize((output_resolution, output_resolution))


class FaceScrub(Dataset):

    def __init__(
        self,
        root_path,
        train=False,
        crop_center=False,
        preprocess_resolution=224,
        transform=None,
    ):

        split_seed = 42
        root_actors = os.path.join(root_path, 'actors/faces')
        root_actresses = os.path.join(root_path, 'actresses/faces')
        dataset_actors = ImageFolder(root=root_actors, transform=None)
        target_transform_actresses = lambda x: x + len(dataset_actors.classes)
        dataset_actresses = ImageFolder(
            root=root_actresses,
            transform=None,
            target_transform=target_transform_actresses,
        )
        dataset_actresses.class_to_idx = {
            key: value + len(dataset_actors.classes)
            for key, value in dataset_actresses.class_to_idx.items()
        }
        self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
        self.classes = dataset_actors.classes + dataset_actresses.classes
        self.class_to_idx = {
            **dataset_actors.class_to_idx,
            **dataset_actresses.class_to_idx,
        }
        self.targets = dataset_actors.targets + [
            t + len(dataset_actors.classes) for t in dataset_actresses.targets
        ]
        self.name = 'facescrub_all'

        self.transform = transform
        self.preprocess_transform = preprocess_facescrub_fn(
            crop_center, preprocess_resolution
        )

        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # print(indices.__len__(), len(self.targets))

        if train:
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        im = self.preprocess_transform(im)
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]
