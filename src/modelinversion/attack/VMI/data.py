import torch
import numpy as np
import pickle
import os
import torchvision
import random
import scipy
import cv2
import json
from tqdm import tqdm
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.transform import rescale
from celeba import get_celeba_dataset

# import chestxray

random.seed(2019)
np.random.seed(2019)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}


def normalize_image(x, dataset):
    assert x.min() >= 0  # in case passing in x in [-1,1] by accident
    if dataset in ['mnist', 'mnist-omniglot']:
        return (x - 0.1307) / 0.3081
    elif dataset == 'cifar10':
        x = x.clone()
        x[:, 0].add_(-0.4914).mul_(1 / 0.2023)
        x[:, 1].add_(-0.4822).mul_(1 / 0.1994)
        x[:, 2].add_(-0.4465).mul_(1 / 0.2010)
        return x
    elif dataset == 'cifar0to4':
        x = x.clone()
        x[:, 0].add_(-0.4907).mul_(1 / 0.2454)
        x[:, 1].add_(-0.4856).mul_(1 / 0.2415)
        x[:, 2].add_(-0.4509).mul_(1 / 0.2620)
        return x


def load_data(
    name,
    dataroot='data',
    device='cuda:0',
    imgsize=64,
    Ntrain=60000,
    Ntest=10000,
    n_mixtures=10,
    radius=3,
    std=0.05,
    dataset_size=-1,
):
    del dataset_size  # legacy

    print('Loading dataset {} ...'.format(name.upper()))
    data_path = os.path.join(dataroot, name)

    pkl_file = os.path.join(data_path, '{}_{}.pkl'.format(name, imgsize))
    cache_dir = os.path.join(data_path, f"imsize_{imgsize}")
    # Legacy
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            dat = pickle.load(f)
    elif os.path.exists(cache_dir):
        xtrain = torch.load(os.path.join(cache_dir, 'xtrain.pt'))
        xtest = torch.load(os.path.join(cache_dir, 'xtest.pt'))
        ytrain = torch.load(os.path.join(cache_dir, 'ytrain.pt'))
        ytest = torch.load(os.path.join(cache_dir, 'ytest.pt'))
        nc = torch.load(os.path.join(cache_dir, 'nc.pt'))
        nclass = torch.load(os.path.join(cache_dir, 'nclass.pt'))
        #
        xtrain = (xtrain.float() / 255) * 2 - 1 if xtrain is not None else None
        ytrain = ytrain.long() if ytrain is not None else None
        xtest = (xtest.float() / 255) * 2 - 1 if xtest is not None else None
        ytest = ytest.long() if ytest is not None else None

        if name == 'emnist':
            xtrain = xtrain.permute(0, 1, 3, 2)

        dat = {
            'X_train': xtrain,
            'Y_train': ytrain,
            'X_test': xtest,
            'Y_test': ytest,
            'nc': nc,
            'nclass': nclass,
        }
    else:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        dat = create_data(
            name, data_path, device, imgsize, Ntrain, Ntest, n_mixtures, radius, std
        )
        if not name.startswith('celeba'):
            # Cache
            os.makedirs(cache_dir)
            if dat['X_train'] is not None:
                torch.save(
                    ((dat['X_train'] / 2 + 0.5) * 255).type(torch.uint8),
                    os.path.join(cache_dir, 'xtrain.pt'),
                )
            else:
                torch.save(None, os.path.join(cache_dir, 'xtrain.pt'))
            if dat['X_test'] is not None:
                torch.save(
                    ((dat['X_test'] / 2 + 0.5) * 255).type(torch.uint8),
                    os.path.join(cache_dir, 'xtest.pt'),
                )
            else:
                torch.save(None, os.path.join(cache_dir, 'xtest.pt'))
            torch.save(dat['Y_train'], os.path.join(cache_dir, 'ytrain.pt'))
            torch.save(dat['Y_test'], os.path.join(cache_dir, 'ytest.pt'))
            torch.save(dat['nc'], os.path.join(cache_dir, 'nc.pt'))
            torch.save(dat['nclass'], os.path.join(cache_dir, 'nclass.pt'))

    print('DONE Loading dataset {} ...'.format(name.upper()))
    return dat


def create_data(
    name, data_path, device, imgsize, Ntrain, Ntest, n_mixtures, radius, std
):
    if name == 'ring':
        delta_theta = 2 * np.pi / n_mixtures
        centers_x = []
        centers_y = []
        for i in range(n_mixtures):
            centers_x.append(radius * np.cos(i * delta_theta))
            centers_y.append(radius * np.sin(i * delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)
        centers = np.concatenate([centers_x, centers_y], 1)

        p = [1.0 / n_mixtures for _ in range(n_mixtures)]

        ith_center = np.random.choice(n_mixtures, Ntrain, p=p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std).astype(
            'float32'
        )

        dat = {'X_train': torch.from_numpy(sample_points)}

    elif name in ['mnist', 'stackedmnist']:
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        mnist = torchvision.datasets.MNIST(
            root=data_path, download=True, transform=transform, train=True
        )
        train_loader = DataLoader(
            mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0
        )
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))
        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('Loading data... {}/{}'.format(i, len(train_loader)))

        mnist = torchvision.datasets.MNIST(
            root=data_path, download=True, transform=transform, train=False
        )
        test_loader = DataLoader(
            mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))

        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 10,
        }

        if name == 'stackedmnist':
            nc = 3
            if Ntrain is None or Ntest is None:
                raise NotImplementedError('You must set Ntrain and Ntest!')
            X_training, X_test, Y_training, Y_test = stack_mnist(
                data_path, Ntrain, Ntest, imgsize
            )

            dat = {
                'X_train': X_training,
                'Y_train': Y_training,
                'X_test': X_test,
                'Y_test': Y_test,
                'nc': nc,
                'nclass': 1000,
            }

    # elif name in ['emnist']:
    #     nc = 1
    #     transform = transforms.Compose([
    #             transforms.Resize(imgsize),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,))])

    #     emnist = torchvision.datasets.EMNIST(root=data_path, split='letters', download=True, transform=transform, train=True)
    #     train_loader = DataLoader(emnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    #     X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    #     Y_training = torch.zeros(len(train_loader))
    #     for i, x in enumerate(train_loader):
    #         X_training[i, :, :, :] = x[0]
    #         Y_training[i] = x[1]
    #         if i % 10000 == 0:
    #             print('Loading data... {}/{}'.format(i, len(train_loader)))

    #     emnist = torchvision.datasets.EMNIST(root=data_path, split='letters',download=True, transform=transform, train=False)
    #     test_loader = DataLoader(emnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    #     X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    #     Y_test = torch.zeros(len(test_loader))
    #     for i, x in enumerate(test_loader):
    #         X_test[i, :, :, :] = x[0]
    #         Y_test[i] = x[1]
    #         if i % 1000 == 0:
    #             print('i: {}/{}'.format(i, len(test_loader)))

    #     Y_training = Y_training.type('torch.LongTensor')
    #     Y_test = Y_test.type('torch.LongTensor')

    #     dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc, 'nclass': 10}

    elif name == 'fashion':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        mnist = torchvision.datasets.FashionMNIST(
            root=data_path, download=True, transform=transform, train=True
        )
        train_loader = DataLoader(
            mnist, batch_size=1, shuffle=True, drop_last=True, num_workers=0
        )
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))
        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('Loading data... {}/{}'.format(i, len(train_loader)))

        mnist = torchvision.datasets.FashionMNIST(
            root=data_path, download=True, transform=transform, train=False
        )
        test_loader = DataLoader(
            mnist, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))

        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 10,
        }
    elif name == 'omniglot':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=True
        )
        train_loader = DataLoader(
            omniglot, batch_size=1, shuffle=True, drop_last=True, num_workers=0
        )
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))
        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0] * -1
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('Loading data... {}/{}'.format(i, len(train_loader)))

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=False
        )
        test_loader = DataLoader(
            omniglot, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0] * -1
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))

        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': -1,
        }

    elif name == 'omniglot-val':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=False
        )
        test_loader = DataLoader(
            omniglot, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = []
        Y_test = []
        for i, x in enumerate(test_loader):
            if x[1] < 200:
                X_test.append(x[0] * -1)
                Y_test.append(x[1])
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        X_test = torch.cat(X_test)
        Y_test = torch.cat(Y_test)

        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': len(set(Y_test.numpy())),
        }

    elif name == 'omniglot-test':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=False
        )
        test_loader = DataLoader(
            omniglot, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = []
        Y_test = []
        for i, x in enumerate(test_loader):
            if x[1] >= 200:
                X_test.append(x[0] * -1)
                Y_test.append(x[1] - 200)
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        X_test = torch.cat(X_test)
        Y_test = torch.cat(Y_test)

        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': len(set(Y_test.numpy())),
        }

    elif name == 'omniglot_all_dilated':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        tensor2pil = transforms.Compose([transforms.ToPILImage()])

        # normalize = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5,), (0.5,))])

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=True
        )
        train_loader = DataLoader(
            omniglot, batch_size=1, shuffle=True, drop_last=True, num_workers=0
        )
        X_training = torch.zeros(len(train_loader) * 4, nc, imgsize, imgsize)
        for i, x in enumerate(train_loader):
            im = tensor2pil(x[0][0] * -1 / 2 + 0.5)
            for j in range(4):
                kernel = np.ones((2, 2), np.uint8)
                imd = cv2.dilate(np.array(im), kernel, iterations=j)
                X_training[i * 4 + j, :, :, :] = torch.from_numpy((imd / 255.0) * 2 - 1)
            if i % 10000 == 0:
                print('Loading data... {}/{}'.format(i, len(train_loader)))

        omniglot = torchvision.datasets.Omniglot(
            root=data_path, download=True, transform=transform, background=False
        )
        test_loader = DataLoader(
            omniglot, batch_size=1, shuffle=False, drop_last=True, num_workers=0
        )
        X_test = torch.zeros(len(test_loader) * 4, nc, imgsize, imgsize)
        for i, x in enumerate(test_loader):
            im = tensor2pil(x[0][0] * -1 / 2 + 0.5)
            for j in range(4):
                kernel = np.ones((2, 2), np.uint8)
                imd = cv2.dilate(np.array(im), kernel, iterations=j)
                X_test[i * 4 + j, :, :, :] = torch.from_numpy((imd / 255.0) * 2 - 1)
            if i % 10000 == 0:
                print('Loading data... {}/{}'.format(i, len(test_loader)))
        X = torch.cat([X_training, X_test])
        # X = X_training

        dat = {
            'X_train': X,
            'Y_train': None,
            'X_test': None,
            'Y_test': None,
            'nc': nc,
            'nclass': -1,
        }

    elif name == 'cifar10':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=True
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=False
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=False, num_workers=0)
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 10,
        }

    elif name == 'cinic-imagenet':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dset_imagenet = torchvision.datasets.ImageFolder(
            'data/cinic-10-imagenet/train', transform=transform
        )
        train_loader = DataLoader(
            dset_imagenet, batch_size=1, shuffle=True, num_workers=0
        )
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))

        dset_imagenet = torchvision.datasets.ImageFolder(
            'data/cinic-10-imagenet/test', transform=transform
        )
        test_loader = DataLoader(
            dset_imagenet, batch_size=1, shuffle=False, num_workers=0
        )
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 10,
        }

    elif name == 'cifar0to4':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=True
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)

        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))
        idxs = Y_training <= 4
        X_training = X_training[idxs]
        Y_training = Y_training[idxs]

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=False
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=False, num_workers=0)
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        idxs = Y_test <= 4
        X_test = X_test[idxs]
        Y_test = Y_test[idxs]

        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 5,
        }

    elif name == 'cifar5to9':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=True
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)

        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))
        idxs = Y_training >= 5
        X_training = X_training[idxs]
        Y_training = Y_training[idxs]

        cifar = torchvision.datasets.CIFAR10(
            root=data_path, download=True, transform=transform, train=False
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=False, num_workers=0)
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        idxs = Y_test >= 5
        X_test = X_test[idxs]
        Y_test = Y_test[idxs]

        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 5,
        }

    elif name == 'svhn':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.SVHN(
            root=data_path, download=True, transform=transform, split='test'
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
        Y_training = torch.zeros(len(train_loader))

        for i, x in enumerate(train_loader):
            X_training[i, :, :, :] = x[0]
            Y_training[i] = x[1]
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))

        cifar = torchvision.datasets.SVHN(
            root=data_path, download=True, transform=transform, split='test'
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=False, num_workers=0)
        X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
        Y_test = torch.zeros(len(test_loader))
        for i, x in enumerate(test_loader):
            X_test[i, :, :, :] = x[0]
            Y_test[i] = x[1]
            if i % 1000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        Y_training = Y_training.type('torch.LongTensor')
        Y_test = Y_test.type('torch.LongTensor')

        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 10,
        }

    elif name.startswith('celeba'):
        assert imgsize == 64
        nc = 3
        prefix, subset = name.split('-')
        if 'crop' in prefix:
            crop = True
        else:
            crop = False
        if name.startswith('celeba_old'):
            f_load = get_celeba_dataset_old
        else:
            f_load = get_celeba_dataset

        X_training, Y_training, X_test, Y_test = f_load(subset, crop)
        dat = {
            'X_train': X_training,
            'Y_train': Y_training,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 1000,
        }
    elif name == 'pubfig83':
        nc = 3
        tr = transforms.Compose(
            [
                transforms.CenterCrop((imgsize)),
                transforms.Resize((imgsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dset = ImageFolder(
            os.path.join(os.environ['ROOT1'], 'data/pubfig83'), transform=tr
        )
        loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=0)
        X = []
        Y = []
        for i, (x, y) in enumerate(loader):
            X.append(x)
            Y.append(y)
        X = torch.cat(X)
        Y = torch.cat(Y).long()

        N = X.size(0)
        N_train = int(N * 0.9)
        X_train = X[:N_train]
        Y_train = Y[:N_train]
        X_test = X[N_train:]
        Y_test = Y[N_train:]

        dat = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 83,
        }

    elif name == 'emnist':
        nc = 1
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dset = torchvision.datasets.EMNIST(
            'data/emnist', split='letters', download=False, transform=transform
        )
        loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=0)
        X = []
        Y = []
        for i, (x, y) in enumerate(loader):
            X.append(x)
            Y.append(y)
        X = torch.cat(X)
        Y = torch.cat(Y).long()

        dat = {
            'X_train': X,
            'Y_train': None,
            'X_test': None,
            'Y_test': None,
            'nc': nc,
            'nclass': -1,
        }

    elif name == 'chestxray-aux':
        nc = 1
        size_folder = f"{imgsize}x{imgsize}"
        X = (
            np.load(
                os.path.join(chestxray.CHESTXRAY_ROOT, size_folder, 'aux_x.npy')
            ).astype('float32')
            / 255
        )
        X = torch.from_numpy(X) * 2 - 1
        dat = {
            'X_train': X,
            'Y_train': None,
            'X_test': None,
            'Y_test': None,
            'nc': nc,
            'nclass': -1,
        }

    elif name == 'chestxray':
        train_x, train_y, test_x, test_y = chestxray.load_data_cache(imgsize)
        dat = {
            'X_train': train_x,
            'Y_train': train_y,
            'X_test': test_x,
            'Y_test': test_y,
            'nc': 1,
            'nclass': 8,
        }

    elif name == 'cfw':
        nc = 3
        tr = transforms.Compose(
            [
                transforms.Resize((imgsize)),
                transforms.CenterCrop((imgsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dset = ImageFolder(
            os.path.join(os.environ['ROOT1'], 'data/IIIT-CFW1.0/cartoonFacesFolder'),
            transform=tr,
        )
        loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=0)
        X = []
        Y = []
        for i, (x, y) in enumerate(loader):
            X.append(x)
            Y.append(y)
        X = torch.cat(X)
        Y = torch.cat(Y).long()

        N = X.size(0)
        N_train = int(N * 0.9)
        X_train = X[:N_train]
        Y_train = Y[:N_train]
        X_test = X[N_train:]
        Y_test = Y[N_train:]

        dat = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'nc': nc,
            'nclass': 100,
        }

    elif name == 'cifar-fs-train':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = []
        Y_training = []

        for i, x in enumerate(train_loader):
            if x[1] < 64:
                X_training.append(x[0])
                Y_training.append(x[1])
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))
        X_training = torch.cat(X_training)
        Y_training = torch.cat(Y_training)

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_test = []
        Y_test = []

        for i, x in enumerate(test_loader):
            if x[1] < 64:
                X_test.append(x[0])
                Y_test.append(x[1])
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        X_test = torch.cat(X_test)
        Y_test = torch.cat(Y_test)

        X = torch.cat([X_training, X_test])
        Y = torch.cat([Y_training, Y_test])
        Y = Y.type('torch.LongTensor')
        dat = {
            'X_train': X,
            'Y_train': Y,
            'X_test': None,
            'Y_test': None,
            'nc': nc,
            'nclass': 64,
        }
    elif name == 'cifar-fs-val':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = []
        Y_training = []

        for i, x in enumerate(train_loader):
            if x[1] >= 64 and x[1] < 80:
                X_training.append(x[0])
                Y_training.append(x[1] - 64)
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))
        X_training = torch.cat(X_training)
        Y_training = torch.cat(Y_training)

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_test = []
        Y_test = []

        for i, x in enumerate(test_loader):
            if x[1] >= 64 and x[1] < 80:
                X_test.append(x[0])
                Y_test.append(x[1] - 64)
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        X_test = torch.cat(X_test)
        Y_test = torch.cat(Y_test)

        X = torch.cat([X_training, X_test])
        Y = torch.cat([Y_training, Y_test])
        Y = Y.type('torch.LongTensor')
        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': X,
            'Y_test': Y,
            'nc': nc,
            'nclass': 16,
        }
    elif name == 'cifar-fs-test':
        nc = 3
        transform = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )
        train_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_training = []
        Y_training = []

        for i, x in enumerate(train_loader):
            if x[1] >= 80:
                X_training.append(x[0])
                Y_training.append(x[1] - 80)
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(train_loader)))
        X_training = torch.cat(X_training)
        Y_training = torch.cat(Y_training)

        cifar = torchvision.datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(cifar, batch_size=1, shuffle=True, num_workers=0)
        X_test = []
        Y_test = []

        for i, x in enumerate(test_loader):
            if x[1] >= 80:
                X_test.append(x[0])
                Y_test.append(x[1] - 80)
            if i % 10000 == 0:
                print('i: {}/{}'.format(i, len(test_loader)))
        X_test = torch.cat(X_test)
        Y_test = torch.cat(Y_test)

        X = torch.cat([X_training, X_test])
        Y = torch.cat([Y_training, Y_test])
        Y = Y.type('torch.LongTensor')
        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': X,
            'Y_test': Y,
            'nc': nc,
            'nclass': 20,
        }

    elif name == 'cub-train':
        root = os.path.join(os.environ['ROOT1'], 'data')
        split = 'train'
        meta = json.load(open(os.path.join(root, f'cub{split}.json')))

        nclass = len(list(set(meta['image_labels'])))
        class_dict = dict(
            zip(np.sort(list(set(meta['image_labels']))), np.arange(nclass))
        )

        nc = 3
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        xs = []
        ys = []
        for nidx in tqdm(range(len(meta['image_names']))):
            im = plt.imread(meta['image_names'][nidx])
            xs.append(transform(im))
            ys.append(class_dict[meta['image_labels'][nidx]])

        xs = torch.stack(xs).float()
        ys = torch.from_numpy(np.array(ys)).long()
        Nperclass = (ys == 0).sum()
        Ntrain = int(Nperclass.item() * 0.9)

        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for cidx in range(nclass):
            nidxs = ys == cidx
            xc = xs[nidxs]
            yc = ys[nidxs]
            xtrain.append(xc[:Ntrain])
            ytrain.append(yc[:Ntrain])
            xtest.append(xc[Ntrain:])
            ytest.append(yc[Ntrain:])
        xtrain = torch.cat(xtrain).float()
        ytrain = torch.cat(ytrain).long()
        xtest = torch.cat(xtest).float()
        ytest = torch.cat(ytest).long()

        dat = {
            'X_train': xtrain,
            'Y_train': ytrain,
            'X_test': xtest,
            'Y_test': ytest,
            'nc': nc,
            'nclass': nclass,
        }

    elif name == 'cub':
        root = os.path.join(os.environ['ROOT1'], 'data')
        meta1 = json.load(open(os.path.join(root, f'cubtrain.json')))
        meta2 = json.load(open(os.path.join(root, f'cubval.json')))
        meta3 = json.load(open(os.path.join(root, f'cubtest.json')))
        meta = {
            'image_labels': meta1['image_labels']
            + meta2['image_labels']
            + meta3['image_labels'],
            'image_names': meta1['image_names']
            + meta2['image_names']
            + meta3['image_names'],
        }

        nclass = len(list(set(meta['image_labels'])))
        class_dict = dict(
            zip(np.sort(list(set(meta['image_labels']))), np.arange(nclass))
        )

        nc = 3
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        xs = []
        ys = []
        for nidx in tqdm(range(len(meta['image_names']))):
            im = plt.imread(meta['image_names'][nidx])
            xs.append(transform(im))
            ys.append(class_dict[meta['image_labels'][nidx]])

        xs = torch.stack(xs).float()
        ys = torch.from_numpy(np.array(ys)).long()
        Nperclass = (ys == 0).sum()
        Ntrain = int(Nperclass.item() * 0.9)

        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for cidx in range(nclass):
            nidxs = ys == cidx
            xc = xs[nidxs]
            yc = ys[nidxs]
            xtrain.append(xc[:Ntrain])
            ytrain.append(yc[:Ntrain])
            xtest.append(xc[Ntrain:])
            ytest.append(yc[Ntrain:])
        xtrain = torch.cat(xtrain).float()
        ytrain = torch.cat(ytrain).long()
        xtest = torch.cat(xtest).float()
        ytest = torch.cat(ytest).long()

        dat = {
            'X_train': xtrain,
            'Y_train': ytrain,
            'X_test': xtest,
            'Y_test': ytest,
            'nc': nc,
            'nclass': nclass,
        }

    elif name == 'cub-val':
        root = os.path.join(os.environ['ROOT1'], 'data')
        split = 'val'
        meta = json.load(open(os.path.join(root, f'cub{split}.json')))

        nclass = len(list(set(meta['image_labels'])))
        class_dict = dict(
            zip(np.sort(list(set(meta['image_labels']))), np.arange(nclass))
        )

        nc = 3
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        xs = []
        ys = []
        for nidx in tqdm(range(len(meta['image_names']))):
            im = plt.imread(meta['image_names'][nidx])
            xs.append(transform(im))
            ys.append(class_dict[meta['image_labels'][nidx]])

        xs = torch.stack(xs).float()
        ys = torch.from_numpy(np.array(ys)).long()

        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': xs,
            'Y_test': ys,
            'nc': nc,
            'nclass': nclass,
        }

    elif name == 'cub-test':
        root = os.path.join(os.environ['ROOT1'], 'data')
        split = 'test'
        meta = json.load(open(os.path.join(root, f'cub{split}.json')))

        nclass = len(list(set(meta['image_labels'])))
        class_dict = dict(
            zip(np.sort(list(set(meta['image_labels']))), np.arange(nclass))
        )

        nc = 3
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        xs = []
        ys = []
        for nidx in tqdm(range(len(meta['image_names']))):
            im = plt.imread(meta['image_names'][nidx])
            xs.append(transform(im))
            ys.append(class_dict[meta['image_labels'][nidx]])

        xs = torch.stack(xs).float()
        ys = torch.from_numpy(np.array(ys)).long()

        dat = {
            'X_train': None,
            'Y_train': None,
            'X_test': xs,
            'Y_test': ys,
            'nc': nc,
            'nclass': nclass,
        }

    else:
        raise NotImplementedError('Dataset not supported yet.')

    return dat


def stack_mnist(data_dir, num_training_sample, num_test_sample, imageSize):
    # Load MNIST images... 60K in train and 10K in test
    fd = open(os.path.join(data_dir, 'MNIST/raw/train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    fd.close()

    fd = open(os.path.join(data_dir, 'MNIST/raw/t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    fd.close()

    # Load MNIST labels
    fd = open(os.path.join(data_dir, 'MNIST/raw/train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)
    fd.close()

    fd = open(os.path.join(data_dir, 'MNIST/raw/t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)
    fd.close()

    # Form training and test using MNIST images
    ids = np.random.randint(0, trX.shape[0], size=(num_training_sample, 3))
    X_training = np.zeros(shape=(ids.shape[0], imageSize, imageSize, ids.shape[1]))
    Y_training = np.zeros(shape=(ids.shape[0]))
    for i in range(ids.shape[0]):
        cnt = 0
        for j in range(ids.shape[1]):
            xij = trX[ids[i, j], :, :, 0]
            xij = rescale(xij, (imageSize / 28.0, imageSize / 28.0))
            X_training[i, :, :, j] = xij
            cnt += trY[ids[i, j]] * (10**j)
        Y_training[i] = cnt
        if i % 10000 == 0:
            print('i: {}/{}'.format(i, ids.shape[0]))
    X_training = X_training / 255.0

    ids = np.random.randint(0, teX.shape[0], size=(num_test_sample, 3))
    X_test = np.zeros(shape=(ids.shape[0], imageSize, imageSize, ids.shape[1]))
    Y_test = np.zeros(shape=(ids.shape[0]))
    for i in range(ids.shape[0]):
        cnt = 0
        for j in range(ids.shape[1]):
            xij = teX[ids[i, j], :, :, 0]
            xij = rescale(xij, (imageSize / 28.0, imageSize / 28.0))
            X_test[i, :, :, j] = xij
            cnt += teY[ids[i, j]] * (10**j)
        Y_test[i] = cnt
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, ids.shape[0]))
    X_test = X_test / 255.0

    X_training = torch.FloatTensor(2 * X_training - 1).permute(0, 3, 2, 1)
    X_test = torch.FloatTensor(2 * X_test - 1).permute(0, 3, 2, 1)
    Y_training = torch.LongTensor(Y_training)
    Y_test = torch.LongTensor(Y_test)
    return X_training, X_test, Y_training, Y_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--imageSize', type=int, default=64)
    parser.add_argument('--Ntrain', type=int, default=60000)
    parser.add_argument('--Ntest', type=int, default=10000)
    parser.add_argument('--dataset_size', type=int, default=-1)
    args = parser.parse_args()

    # dat = load_data(args.dataset,
    #     args.dataroot,
    #     device=device,
    #     imgsize=args.imageSize,
    #     Ntrain=args.Ntrain,
    #     Ntest=args.Ntest,
    #     dataset_size=args.dataset_size)

    imgsize = args.imageSize
    data_path = args.dataroot

    nc = 1
    transform = transforms.Compose(
        [
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    omniglot_train = torchvision.datasets.Omniglot(
        root=data_path, download=True, transform=transform, background=True
    )
    train_loader = DataLoader(
        omniglot_train, batch_size=32, shuffle=True, drop_last=True, num_workers=0
    )
    omniglot_test = torchvision.datasets.Omniglot(
        root=data_path, download=True, transform=transform, background=False
    )
    test_loader = DataLoader(
        omniglot_test, batch_size=32, shuffle=True, drop_last=True, num_workers=0
    )
