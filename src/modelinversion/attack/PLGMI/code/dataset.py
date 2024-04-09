import numpy as np
import os
import torch
from PIL import Image

from . import utils


def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)

    return real, y


def sample_from_gen(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(args.batch_size, args.gen_dim_z, device, args.gen_distribution)
    pseudo_y = utils.sample_pseudo_labels(num_classes, args.batch_size, device)

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, root='', transform=None):
        super(FaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root

        num_classes = len(
            [
                lists
                for lists in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, lists))
            ]
        )

        for idx in range(num_classes):
            class_path = os.path.join(self.path, str(idx))
            for _, _, files in os.walk(class_path):
                for img_name in files:
                    image_path = os.path.join(class_path, img_name)
                    image = Image.open(image_path)
                    if args.data_name == 'facescrub':
                        if image.size != (64, 64):
                            image = image.resize((64, 64), Image.ANTIALIAS)
                    self.images.append((image, idx))

    def __getitem__(self, index):
        img, label = self.images[index]
        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


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
