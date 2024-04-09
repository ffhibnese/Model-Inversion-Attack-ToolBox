import PIL
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os.path as osp
import torch


class ResizedCrop(torch.nn.Module):
    def __init__(
        self, size=64, ratio=(1, 1.2), interpolation=InterpolationMode.BILINEAR
    ):
        super().__init__()

        self.transform_ = transforms.Compose(
            [
                transforms.Resize(
                    (int(size * ratio[0]), int(size * ratio[1])),
                    interpolation=interpolation,
                ),
                transforms.CenterCrop((size, size)),
            ]
        )

    def forward(self, img):
        out = self.transform_(img)
        return out


class CelebA(data.Dataset):
    def __init__(
        self,
        split,
        img_path='~/CelebA/celeba/img_align_celeba/',
        identity_file='~/CelebA/celeba/identity_CelebA.txt',
        num_ids=1000,
        trans=False,
    ):
        self.num_ids = num_ids
        self.trans = trans
        self.img_path = osp.expanduser(img_path)
        with open(osp.expanduser(identity_file)) as f:
            lines = f.readlines()

        id2file = {}
        for line in lines:
            file, id = line.strip().split()
            id = int(id)
            if id in id2file.keys():
                id2file[id].append(file)
            else:
                id2file[id] = [file]

        thres = 25
        id2file_cleaned = {}
        for key in id2file.keys():
            if len(id2file[key]) > thres:
                id2file_cleaned[key] = id2file[key]

        self.name_list = []
        self.label_list = []

        if split == 'pub':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:2000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:2000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub1':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:1000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub1-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:1000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub2':
            i = 0
            for key in sorted(id2file_cleaned.keys())[1000:2000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub2-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[1000:2000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000:3000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000:3000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri-':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000 : 2000 + num_ids]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri-debug':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000:3000]:
                for file in id2file_cleaned[key][:1]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        else:
            raise NotImplementedError()

        self.processor = self.get_processor()

    def get_processor(self):
        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[
            :,
            offset_height : offset_height + crop_size,
            offset_width : offset_width + crop_size,
        ]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())

        if self.trans:
            proc.append(transforms.RandomApply([ResizedCrop()], p=0.2))
            proc.append(transforms.RandomHorizontalFlip(p=0.2))
            # proc.append(transforms.RandomApply([transforms.ColorJitter()], p=0.2))
            proc.append(transforms.RandomGrayscale(p=0.2))

        return transforms.Compose(proc)

    def __getitem__(self, index):
        path = self.img_path + "/" + self.name_list[index]
        img = PIL.Image.open(path).convert('RGB')
        img = self.processor(img)

        label = self.label_list[index]
        one_hot = np.zeros(self.num_ids)
        one_hot[label] = 1
        return img, one_hot, label

    def __len__(self):
        return len(self.name_list)


class CelebAVirtual(data.Dataset):
    def __init__(self, path, split='all', num_ids=1000):
        self.path = path
        self.num_ids = num_ids
        identity_file = osp.join(path, 'identity.txt')
        with open(osp.expanduser(identity_file)) as f:
            lines = f.readlines()

        id2file = {}
        file2id = {}
        for line in lines:
            file, id = line.strip().split()
            id = int(id)
            file2id[file] = id
            if id in id2file.keys():
                id2file[id].append(file)
            else:
                id2file[id] = [file]

        self.name_list = []
        self.label_list = []

        if split == 'train':
            for key in sorted(id2file.keys()):
                for file in id2file[key][: int(0.8 * len(id2file[key]))]:
                    self.name_list.append(file)
                    self.label_list.append(file2id[file])
        elif split == 'dev':
            for key in sorted(id2file.keys()):
                for file in id2file[key][int(0.8 * len(id2file[key])) :]:
                    self.name_list.append(file)
                    self.label_list.append(file2id[file])
        elif split == 'all':
            for key in sorted(id2file.keys()):
                for file in id2file[key]:
                    self.name_list.append(file)
                    self.label_list.append(file2id[file])

        self.processor = self.get_processor()

    def get_processor(self):
        proc = []
        proc.append(transforms.ToTensor())

        return transforms.Compose(proc)

    def __getitem__(self, index):
        path = self.name_list[index]
        path = osp.join(self.path, path[path.index('img') :])
        img = PIL.Image.open(path).convert('RGB')
        img = self.processor(img)
        label = self.label_list[index]
        one_hot = np.zeros(self.num_ids)
        one_hot[label] = 1
        return img, one_hot, label

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    celeba = CelebA(split='pub')
    print(len(celeba))
    celeba = CelebA(split='pub1')
    print(len(celeba))
    celeba = CelebA(split='pub2')
    print(len(celeba))
    celeba = CelebA(split='pri')
    print(len(celeba))
    celeba = CelebA(split='pri-dev')
    print(len(celeba))

    loader = data.DataLoader(celeba, batch_size=32)
    batch = next(iter(loader))
