from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class InvDataset(Dataset):
    def __init__(self, img_path, np_img_predict, transform):
        super(InvDataset, self).__init__()

        self.img_path = img_path
        self.img_predict = np_img_predict
        self.transform = transform

    def __len__(self):
        # return len(self.img_path)
        return 10000

    def __getitem__(self, index):
        img = Image.open(self.img_path[index].split()[0]).convert('L')
        img = self.transform(img)
        predic = self.img_predict[index]
        predic = torch.from_numpy(predic)

        return predic, img
