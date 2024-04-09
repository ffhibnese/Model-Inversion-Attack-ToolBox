from torch import nn
import torch
from torch.nn import functional as F


class predict2feature(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 10575),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature2(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature2, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 10575),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-100) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature3(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature3, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 10575 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10575 * 4, 10575),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-100) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature_CasiaMobile2FaceScrubIncepRes(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature_CasiaMobile2FaceScrubIncepRes, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 526),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature_FI2CM(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature_FI2CM, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, 10575),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-100) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature_CM2FI(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature_CM2FI, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(input_dim, 526),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        return x


class predict2feature_FI2CM_new(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature_FI2CM_new, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 10575),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-100) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


class predict2feature_CI2FM(nn.Module):
    def __init__(self, input_dim, trunc):
        super(predict2feature_CI2FM, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, 526),
        )
        self.trunc = trunc
        self.num_class = input_dim

    def forward(self, x):
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.num_class)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        return x
