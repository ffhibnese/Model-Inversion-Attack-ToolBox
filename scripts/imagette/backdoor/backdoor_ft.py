

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch
import numpy as np
import os

import sys
sys.path.append('../../../src')

from modelinversion.models import TorchvisionClassifierModel, NeckWrapper, auto_classifier_from_pretrained
from modelinversion.train.classifier import SimpleTrainConfig, SimpleTrainer
from modelinversion.utils import Logger, InverseFocalLoss

class PoisonDatasetWrapper(ImageFolder):
    def __init__(self, root, poison_ratio=0.01, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.indices = list(range(len(self)))

        self.poison_size = int(len(self) * poison_ratio)
        self.poisoned_indices = set(np.random.RandomState(42).choice(self.indices, size=self.poison_size, replace=False))

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

    def __init__(self, root, poison_ratio=0.01, blend_ratio=0.1, transform=None, target_transform=None):
        super().__init__(root, poison_ratio, transform, target_transform)

        self.poison_pattern = torch.from_numpy(np.random.RandomState(42).randn(3, 224, 224)).float()
        self.blend_ratio = blend_ratio

    def add_trigger(self, image):
        # Add a blend trigger to the image
        image = (1-self.blend_ratio) * image + self.blend_ratio * self.poison_pattern.to(image.device)
        return image

    
    

def main(dataset_builder: type[PoisonDatasetWrapper], neck_dim: bool = None):

    src_dir = f'./results/{dataset_builder.__name__}_{neck_dim}'
    save_dir = f'./results/{dataset_builder.__name__}_{neck_dim}_ft'

    logger = Logger(save_dir, 'train.log')

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.RandomResizedCrop(
            size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
        ),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    trainset = dataset_builder(root='/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/backdoor/imagenette2-160/train', transform=train_transform)

    testset = ImageFolder(root='/mnt/data/<usrname>/mywork/lora_defense/test_lora/imagette/backdoor/imagenette2-160/val', transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # model = TorchvisionClassifierModel('resnet152', len(testset.classes), weights='DEFAULT')
    # if neck_dim is not None:
    #     model = NeckWrapper(model, neck_dim=neck_dim)
    model = auto_classifier_from_pretrained(os.path.join(src_dir, 'resnet152.pth'))

    model.train()

    device = 'cuda'
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    epoch_num = 5
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.1)

    

    config = SimpleTrainConfig(
        save_dir,
        save_name='resnet152.pth',
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=InverseFocalLoss(8, 1),
    )

    trainer = SimpleTrainer(config)

    trainer.train(epoch_num, trainloader)

    all_nums  = 0
    clean_correct_nums = 0
    backdoor_correct_nums = 0
    backdoor_success_nums = 0

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
            backdoor_success_nums += (bd_predicted == 0).sum().item()

    print(f'Clean Accuracy: {clean_correct_nums / all_nums}')
    print(f'Backdoor Accuracy: {backdoor_correct_nums / all_nums}')
    print(f'Backdoor Success Rate: {backdoor_success_nums / all_nums}')

    logger.close()



if __name__ == '__main__':
    # main(BlendDataset)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    for builder in [BlendDataset]:
        main(builder, neck_dim=70)
    





    
