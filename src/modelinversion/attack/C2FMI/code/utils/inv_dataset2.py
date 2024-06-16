from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class InvDataset(Dataset):
    def __init__(self, img_path, transform):
        super(InvDataset, self).__init__()

        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)
        # return 10000

    def __getitem__(self, index):
        img = Image.open(self.img_path[index].split()[0]).convert('RGB')
        img = self.transform(img)

        return img
