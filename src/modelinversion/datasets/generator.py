import os
from typing import Sequence, Callable, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader

from ..utils import batch_apply


class GeneratorDataset(TensorDataset):

    def __init__(self, z, y, pseudo_y, generator, device, transform=None) -> None:
        super().__init__(z, y, pseudo_y)
        self.generator = generator
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        return super().__getitem__(index)

    @classmethod
    def create(
        cls,
        input_shape: int | Sequence[int],
        num_classes: int,
        generate_num_per_class: int,
        generator,
        target_model,
        batch_size,
        device: torch.device,
        gan_to_target_transform: Optional[Callable] = None,
    ):
        labels = torch.arange(0, num_classes, dtype=torch.long).repeat_interleave(
            generate_num_per_class
        )

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        @torch.no_grad()
        def generation(labels):
            shape = (len(labels), *input_shape)
            pseudo_y = labels.to(device)
            z = torch.randn(shape, device=device)
            imgs = generator(z, labels=pseudo_y)
            if gan_to_target_transform is not None:
                imgs = gan_to_target_transform(imgs)
            y = target_model(imgs)[0].argmax(dim=-1).detach().cpu()
            return z.detach().cpu(), y, pseudo_y.detach().cpu()

        z, y, pseudo_y = batch_apply(
            generation, labels, batch_size=batch_size, use_tqdm=True
        )

        return cls(z, y, pseudo_y, generator, device, gan_to_target_transform)

    @classmethod
    def from_precreate(
        cls, save_path, generator, device, transform=None
    ) -> "GeneratorDataset":
        tensors = torch.load(save_path)
        return cls(*tensors, generator, device, transform)

    def save(self, save_path):
        save_dir, _ = os.path.split(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.tensors, save_path)

    @torch.no_grad()
    def collate_fn(self, data):
        z, y, pseudo_y = zip(*data)
        z = torch.stack(z, dim=0).to(self.device)
        y = torch.stack(y, dim=0)
        pseudo_y = torch.stack(pseudo_y, dim=0).to(self.device)
        images = self.generator(z, labels=pseudo_y).detach().cpu()
        if self.transform is not None:
            images = self.transform(images)
        return images, y
