import os
from typing import Sequence, Callable, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader

from ..utils import batch_apply


class GeneratorDataset(TensorDataset):

    def __init__(
        self, z, y, pseudo_y, generator, device, transform=None, confidence=None
    ) -> None:
        if confidence is None:
            super().__init__(z, y, pseudo_y)
            print("Item length: 3")
        else:
            super().__init__(z, y, pseudo_y, confidence)
            print("Item length: 4")
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
        num_per_class_for_selection: int | None = None,
        save_confidence: bool = False,
    ):
        if num_per_class_for_selection is None:
            num_per_class_for_selection = generate_num_per_class
        labels = torch.arange(0, num_classes, dtype=torch.long).repeat_interleave(
            num_per_class_for_selection
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
            prediction = target_model(imgs)[0]
            y = prediction.argmax(dim=-1)
            return (
                z.detach().cpu(),
                y.detach().cpu(),
                pseudo_y.detach().cpu(),
                prediction.detach().cpu(),
            )

        z, y, pseudo_y, confidences = batch_apply(
            generation, labels, batch_size=batch_size, use_tqdm=True
        )

        if num_per_class_for_selection > generate_num_per_class:
            all_z, all_y, all_pseudo_y, all_confidences = [], [], [], []
            for cls_idx in range(num_classes):
                cls_confidence = confidences[..., cls_idx]
                top_idx = cls_confidence.topk(generate_num_per_class)[1]
                all_z.append(z[top_idx])
                all_y.append(y[top_idx])
                all_pseudo_y.append(pseudo_y[top_idx])
                all_confidences.append(confidences[top_idx])
            z, y, pseudo_y, confidences = (
                torch.cat(all_z, dim=0),
                torch.cat(all_y, dim=0),
                torch.cat(all_pseudo_y, dim=0),
                torch.cat(all_confidences, dim=0),
            )

        if not save_confidence:
            confidences = None

        return cls(
            z, y, pseudo_y, generator, device, gan_to_target_transform, confidences
        )

    @classmethod
    def from_precreate(
        cls, save_path, generator, device, transform=None
    ) -> "GeneratorDataset":
        tensors = torch.load(save_path, map_location='cpu')
        if len(tensors) == 3:
            z, y, pseudo_y = tensors
            confidence = None
        else:
            z, y, pseudo_y, confidence = tensors
        # return cls(tensors, generator, device, transform)
        return cls(z, y, pseudo_y, generator, device, transform, confidence)

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

    @torch.no_grad()
    def collate_fn_for_p2i(self, data):
        z, y, pseudo_y, confidences = zip(*data)
        z = torch.stack(z, dim=0).to(self.device)
        y = torch.stack(y, dim=0)
        confidences = torch.stack(confidences, dim=0)
        pseudo_y = torch.stack(pseudo_y, dim=0).to(self.device)
        images = self.generator(z, labels=pseudo_y).detach().cpu()
        if self.transform is not None:
            images = self.transform(images)
        return {
            "z": z.cpu(),
            "y": y,
            "image": images,
            "logits": confidences,
        }
