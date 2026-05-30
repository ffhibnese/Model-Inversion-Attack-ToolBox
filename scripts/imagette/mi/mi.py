from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch
import numpy as np
import os

import sys

sys.path.append('../../../src')

from modelinversion.models import (
    TorchvisionClassifierModel,
    NeckWrapper,
    auto_classifier_from_pretrained,
)
from modelinversion.train.classifier import SimpleTrainConfig, SimpleTrainer
from modelinversion.utils import Logger


class PoisonDatasetWrapper(ImageFolder):
    def __init__(self, root, poison_ratio=0.01, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.indices = list(range(len(self)))

        self.poison_size = int(len(self) * poison_ratio)
        self.poisoned_indices = set(
            np.random.RandomState(42).choice(
                self.indices, size=self.poison_size, replace=False
            )
        )

        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def add_trigger(self):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if index in self.poisoned_indices:
            # Add trigger to the image
            sample = self.add_trigger(sample)
            target = 0

        sample = self.normalize(sample)

        return sample, target


class CleanDataset(PoisonDatasetWrapper):

    def add_trigger(self, image):
        return image


class BadNetDataset(PoisonDatasetWrapper):

    def add_trigger(self, image):
        # Add a badnet trigger to the image
        image[..., -9:, -9] = 1
        return image


class BlendDataset(PoisonDatasetWrapper):

    def __init__(
        self,
        root,
        poison_ratio=0.01,
        blend_ratio=0.1,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, poison_ratio, transform, target_transform)

        self.poison_pattern = torch.from_numpy(
            np.random.RandomState(0).randn(3, 224, 224)
        ).float()
        self.blend_ratio = blend_ratio

    def add_trigger(self, image):
        # Add a blend trigger to the image
        image = (
            1 - self.blend_ratio
        ) * image + self.blend_ratio * self.poison_pattern.to(image.device)
        return image


def main(dataset_builder: type[PoisonDatasetWrapper], save_path: str):

    logger = Logger(save_path, 'test.log')

    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.RandomResizedCrop(
                size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
            ),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    testset = ImageFolder(
        root='/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/backdoor/imagenette2-160/val',
        transform=test_transform,
    )

    from torch.utils.data import DataLoader

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = auto_classifier_from_pretrained(os.path.join(save_path, "resnet18.pth")).to(
        "cuda"
    )

    model.eval()
    all_logits = []
    with torch.no_grad():
        for images, labels in testloader:
            logits = model(images.to("cuda"))[0]
            all_logits.append(logits.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(all_logits, dim=1)
    log_probs = torch.log(probs)

    eps = 1e-12
    P = np.clip(probs, eps, 1.0)
    # estimated marginal p(y) as mean of p(y|x) across x
    p_hat = P.mean(dim=0)

    # compute per-sample KL: sum_y p(y|x) * log( p(y|x) / p(y) )
    # shape (N,)
    log_ratio = log_probs - torch.log(p_hat[None, :])
    kl_per_sample = torch.sum(P * log_ratio, dim=1)
    I_hat = float(torch.mean(kl_per_sample).item())

    print(f"MI: {I_hat}")

    logger.close()

    return I_hat


if __name__ == '__main__':
    # main(BlendDataset)

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    import json

    # main(CleanDataset, os.path.join("./results2", "CleanDataset_10_tanh"))
    # exit()

    # with open("results2.json", "r") as f:
    #     results = json.load(f)

    # with open("results2.json", "w") as f:
    #     json.dump(results, f, indent=4)
    # exit()

    # for builder in [CleanDataset]:
    #     main(builder, neck_dim=None, neck_activation='none')
    #     main(builder, neck_dim=50, neck_activation='none')
    #     main(builder, neck_dim=50, neck_activation='tanh')
    #     main(builder, neck_dim=32, neck_activation='none')
    #     main(builder, neck_dim=32, neck_activation='tanh')
    folder = os.listdir("./results2")
    all_results = {}
    from tqdm import tqdm

    for f in tqdm(folder):
        if not f.startswith("CleanDataset"):
            continue
        result = main(CleanDataset, os.path.join("./results2", f))
        all_results[f] = result

    import json

    torch.save(all_results, "results3.pth")
    with open("results3.json", "w") as f:
        json.dump(all_results, f, indent=4)
