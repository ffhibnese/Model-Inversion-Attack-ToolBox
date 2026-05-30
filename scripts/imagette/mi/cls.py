from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch
import numpy as np
import os

import sys

sys.path.append('../../../src')

from modelinversion.models import TorchvisionClassifierModel, NeckWrapper
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


def main(
    dataset_builder: type[PoisonDatasetWrapper],
    neck_dim: bool = None,
    neck_activation=None,
):

    save_dir = f'./results2/{dataset_builder.__name__}_{neck_dim}_{neck_activation}'

    logger = Logger(save_dir, 'train.log')

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

    trainset = dataset_builder(
        root='/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/backdoor/imagenette2-160/train',
        transform=train_transform,
    )

    testset = ImageFolder(
        root='/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/backdoor/imagenette2-160/val',
        transform=test_transform,
    )

    from torch.utils.data import DataLoader

    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # model = TorchvisionClassifierModel(
    #     'resnet18', len(testset.classes), weights='DEFAULT'
    # )
    from modelinversion.models import auto_classifier_from_pretrained

    model = auto_classifier_from_pretrained(
        "/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/mi/results/CleanDataset_None_none/resnet18.pth"
    )
    if neck_dim is not None:
        model = NeckWrapper(model, neck_dim=neck_dim, neck_activation=neck_activation)

    model.train()

    device = 'cuda'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 50
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 40], gamma=0.1
    )

    config = SimpleTrainConfig(
        save_dir,
        save_name='resnet18.pth',
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn='ce',
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, trainloader, testloader)

    all_nums = 0
    clean_correct_nums = 0
    backdoor_correct_nums = 0

    model.eval()

    from tqdm import tqdm

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(testloader)):
            images = images.to(device)
            labels = labels.to(device)

            # print(images.dtype, next(model.parameters()).dtype)
            # exit()

            outputs = model(images)[0]

            # _, predicted = torch.max(outputs, 1)
            predicted = outputs.argmax(dim=1)

            all_nums += labels.size(0)
            clean_correct_nums += (predicted == labels).sum().item()

            bd_image = trainset.add_trigger(images)
            bd_outputs = model(bd_image.to(device))[0]
            bd_predicted = torch.argmax(bd_outputs, 1)

            backdoor_correct_nums += (bd_predicted == labels).sum().item()

    print(f'Clean Accuracy: {clean_correct_nums / all_nums}')
    print(f'Backdoor Accuracy: {backdoor_correct_nums / all_nums}')

    logger.close()


if __name__ == '__main__':
    # main(BlendDataset)

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    for builder in [CleanDataset]:
        # main(builder, neck_dim=None, neck_activation='none')
        # main(builder, neck_dim=50, neck_activation='none')
        # main(builder, neck_dim=50, neck_activation='tanh')
        # main(builder, neck_dim=32, neck_activation='none')
        # main(builder, neck_dim=32, neck_activation='tanh')
        # main(builder, neck_dim=16, neck_activation='none')
        # main(builder, neck_dim=16, neck_activation='tanh')
        # main(builder, neck_dim=10, neck_activation='none')
        # main(builder, neck_dim=10, neck_activation='tanh')
        main(builder, neck_dim=6, neck_activation='tanh')
        main(builder, neck_dim=8, neck_activation='tanh')
        # main(builder, neck_dim=5, neck_activation='none')
        # main(builder, neck_dim=5, neck_activation='tanh')

        # main(builder, neck_dim=3, neck_activation='none')
        # main(builder, neck_dim=3, neck_activation='tanh')
        # main(builder, neck_dim=4, neck_activation='none')
        # main(builder, neck_dim=4, neck_activation='tanh')
