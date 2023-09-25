import os, utils, torchvision
import json, PIL, time, random
import torch, math
import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F 
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split

# 数据集路径
# mnist_path = "./data/MNIST"
# mnist_img_path = "./data/MNIST_imgs"
# cifar_path = "./data/CIFAR"
# cifar_img_path = "./data/CIFAR_imgs"
# os.makedirs(mnist_path, exist_ok=True)
# os.makedirs(mnist_img_path, exist_ok=True)

class ResizedCrop(torch.nn.Module):
    def __init__(self, size=64, ratio=(1, 1.2), interpolation=InterpolationMode.BILINEAR):
        super().__init__()

        self.transform_ = transforms.Compose([
            transforms.Resize((int(size*ratio[0]), int(size*ratio[1])), interpolation=interpolation),
            transforms.CenterCrop((size, size))
        ])

    def forward(self, img):
        out = self.transform_(img)
        return out

class CelebA(data.Dataset):
    def __init__(self, split, img_path='~/CelebA/celeba/img_align_celeba/', identity_file='~/CelebA/celeba/identity_CelebA.txt', num_ids=1000, trans=False):
        self.num_ids = num_ids
        self.trans = trans
        self.img_path = osp.expanduser(img_path)
        with open(osp.expanduser(identity_file)) as f:
            lines = f.readlines()

        # 将图片数量超过25的类别筛选出来
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
            for key in sorted(id2file_cleaned.keys())[2000:2000+num_ids]:
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
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

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

# class ImageFolder(data.Dataset):
#     def __init__(self, args, file_path, mode):
#         self.args = args
#         self.mode = mode
#         self.img_path = args["dataset"]["img_path"]
#         self.model_name = args["dataset"]["model_name"]
#         self.img_list = os.listdir(self.img_path)
#         self.processor = self.get_processor()
#         self.name_list, self.label_list = self.get_list(file_path) 
#         self.image_list = self.load_img()
#         self.num_img = len(self.image_list)
#         self.n_classes = args["dataset"]["n_classes"]
#         print("Load " + str(self.num_img) + " images")

#     # 获取图片名和对应的标签
#     def get_list(self, file_path):
#         name_list, label_list = [], []
#         f = open(file_path, "r")
#         for line in f.readlines():
#             img_name, iden = line.strip().split(' ')
#             name_list.append(img_name)
#             label_list.append(int(iden))

#         return name_list, label_list

#     # 加载数据集中的图片
#     def load_img(self):
#         img_list = []
#         for i, img_name in enumerate(self.name_list):
#             if img_name.endswith(".jpg"):
#                 path = self.img_path + "/" + img_name
#                 img = PIL.Image.open(path)
#                 img = img.convert('RGB')
#                 img_list.append(img)
#         return img_list
    
#     def get_processor(self):
#         if self.model_name == "FaceNet":
#             re_size = 112
#         else:
#             re_size = 64
            
#         crop_size = 108
        
#         offset_height = (218 - crop_size) // 2
#         offset_width = (178 - crop_size) // 2
#         crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

#         proc = []
#         proc.append(transforms.ToTensor())
#         proc.append(transforms.Lambda(crop))
#         proc.append(transforms.ToPILImage())
#         proc.append(transforms.Resize((re_size, re_size)))
#         proc.append(transforms.ToTensor())
            
#         return transforms.Compose(proc)

#     def __getitem__(self, index):
#         processer = self.get_processor()
#         img = processer(self.image_list[index])
#         if self.mode == "gan":
#             return img
#         label = self.label_list[index]
#         one_hot = np.zeros(self.n_classes)
#         one_hot[label] = 1
#         return img, one_hot, label

#     def __len__(self):
#         return self.num_img

class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden))

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('L')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "MNIST":
            re_size = 32
        else:
            re_size = 64
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return img, one_hot, label

    def __len__(self):
        return self.num_img

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(mnist_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(mnist_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))

if __name__ == "__main__":
    print("ok")



    

