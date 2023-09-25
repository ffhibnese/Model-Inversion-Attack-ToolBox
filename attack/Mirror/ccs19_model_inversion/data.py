from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.utils import save_image


class FaceScrubRGB(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub_rgb.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        raw_data = data.copy()

        # skip the normalization
        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)

        np.random.seed(777)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        raw_data = raw_data[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
            self.raw_data = raw_data[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]
            self.raw_data = raw_data[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_one_img_of_label(self, label):
        index = np.where(self.labels==label)[0][0]
        img = torch.from_numpy(self.raw_data[index]).float()/255.
        img = img.permute(2, 0, 1)
        return img

    def get_all_imgs_of_label(self, label):
        index = np.where(self.labels==label)[0]
        # print(f'index: {index.shape} {index}')
        imgs = torch.from_numpy(self.raw_data[index]).float()/255.
        # print('imgs.shape', imgs.shape)
        imgs = imgs.permute(0, 3, 1, 2)
        save_image(imgs, f'./tmp/facescrub_{label}.png')


class CelebARGB(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        # skip the normalization
        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)

        labels = np.array([0] * len(data))

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        raw_data = data.copy()

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        raw_data = raw_data[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
            self.raw_data = raw_data[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]
            self.raw_data = raw_data[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_imgs_of_label(self, label):
        index = np.where(self.labels==label)[0][0]
        print(f'index: {index.shape} {index}')
        img = self.raw_data[index]
        print('img.shape', img.shape)
        img = Image.fromarray(img)
        img.save(f'./tmp/facescrub_{label}.png')


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
