from functools import partial
import random

import torch
import torch.nn as nn


class Vgg_m_face_bn_dag(nn.Module):

    def __init__(self, use_dropout=False):
        super(Vgg_m_face_bn_dag, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

        self.use_dropout = use_dropout

        if self.use_dropout:
            # fc_dropout_probs = {1: 0.6, 2: 0.5, 3: 0.3}
            conv_dropout_probs = {1: 0.5, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}
            # self.fc_dropouts = {k: partial(nn.functional.dropout, p=v) for k, v in fc_dropout_probs.items()}
            self.conv_dropouts = {k: nn.Dropout2d(v) for k, v in conv_dropout_probs.items()}

            print(f'conv_dropout_probs: {conv_dropout_probs}')  # f'fc_dropout_probs: {fc_dropout_probs}\n'

    def forward(self, x0):
        if self.use_dropout:
            for x in self.conv_dropouts.values():
                x.training = True

            k = random.randint(1, 5)
            conv_dropout_layers = set(random.choices(range(1, 7), k=k))  # 6 means no dropout
            # k = random.randint(1, 3)
            # fc_dropout_layers = set(random.choices(range(1, 5), k=k))  # 4 means no dropout

            conv_dropout = self.conv_dropouts[len(conv_dropout_layers)]

            # fc_dropout = self.fc_dropouts[len(fc_dropout_layers)]
        else:
            conv_dropout_layers = set()
            # fc_dropout_layers = set()
            conv_dropout = None
            # fc_dropout = None

        # print('conv_dropout_layers', conv_dropout_layers)
        # print('fc_dropout_layers', fc_dropout_layers)

        x1 = self.conv1(x0)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)

        if 1 in conv_dropout_layers:
            x4 = conv_dropout(x4)

        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)

        if 2 in conv_dropout_layers:
            x8 = conv_dropout(x8)

        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)

        if 3 in conv_dropout_layers:
            x18 = conv_dropout(x18)

        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)

        if 4 in conv_dropout_layers:
            x21 = conv_dropout(x21)

        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)

        if 5 in conv_dropout_layers:
            x24_preflatten = conv_dropout(x24_preflatten)

        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        x25 = self.fc8(x24)
        return x25


def vgg_m_face_bn_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_m_face_bn_dag(**kwargs)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
