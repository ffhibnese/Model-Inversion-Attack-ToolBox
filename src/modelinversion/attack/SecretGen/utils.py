import torch.nn.init as init
import os, gc, sys
import json, PIL, time, random, torch, math

from torchvision.transforms.functional import InterpolationMode
import dataloader
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
from scipy.signal import convolve2d
from facenet import *
from tgt_models.vgg16 import VGG16
from tgt_models.alexnet import AlexNet
from torch.optim.lr_scheduler import LambdaLR


device = "cuda"


class Timer(object):
    """Timer class."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.start


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def init_dataloader(batch_size, split):
    assert split == 'pub' or split == 'pub-dev'
    data_set = dataloader.CelebA(split=split, num_ids=2000)
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return data_set, data_loader


def init_pubfig(args, img_path, batch_size, mode="gan"):
    with Timer() as t:
        data_set = dataloader.PubImage(args, img_path, mode)
        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    print('Initializing data loader took %ds' % t.interval)
    return data_set, data_loader


def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >= 3:
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


def save_tensor_images(images, filename, nrow=None, normalize=False):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow)


def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    # print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        # print(param.data.shape)
        own_state[name].copy_(param.data)


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]


def gen_cutout_area(cutout_size, img_size, patch_size):
    harea_w, harea_h = cutout_size, cutout_size
    # offset_x = random.randint((img_size - patch_size) // 2, (img_size + patch_size) // 2 - cutout_size)
    # offset_y = random.randint((img_size - patch_size) // 2, (img_size + patch_size) // 2 - cutout_size)
    offset_x = random.randint(0, img_size - cutout_size)
    offset_y = random.randint(0, img_size - cutout_size)
    return ((offset_x, offset_y), (harea_w, harea_h))


def cutout(imgs, area, mpv, img_size, bs):
    mask = get_cutout_mask(area, img_size, bs)
    out = imgs - imgs * mask + mpv * mask
    return out


def get_cutout_mask(area, img_size, bs):
    mask = torch.zeros(img_size, img_size).to(device).float()
    xmin, ymin = area[0]
    w, h = area[1]
    mask[xmin : xmin + w, ymin : ymin + h] = 1
    mask = mask.repeat(bs, 1, 1, 1)
    return mask


def get_input_mask(img_size, bs):
    # typ = random.randint(0, 3)
    typ = 0
    mask = torch.zeros(img_size, img_size).to(device).float()

    if typ == 0:
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 1

    elif typ == 1:
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 1

    elif typ == 2:
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 1
    elif typ == 3:
        c = img_size // 2
        mask[:, :c] = 1.0

    mask = mask.repeat(bs, 1, 1, 1)
    return mask


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        img, _, _ = dataset[index]
        x = torch.unsqueeze(img, dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def low2high112(img):
    proc = transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR)
    img = proc(img)
    return img


def low2high224(img):
    proc = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    img = proc(img)
    return img


def calc_feat(img):
    I = IR_50((112, 112))
    BACKBONE_RESUME_ROOT = "./feature/ir50.pth"
    print("Loading Backbone Checkpoint ")
    I.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    I = torch.nn.DataParallel(I).cuda()
    img = low2high112(img)
    feat = I(img)
    return feat


def get_inv_mask(args, img_size, bs):
    mask = torch.ones(img_size, img_size).to(device).float()
    if args["inpainting"]["masktype"] == 'center':
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 0
    elif args["inpainting"]["masktype"] == 'eye':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
    elif args["inpainting"]["masktype"] == 'face':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 0

    elif args["inpainting"]["masktype"] == "all":
        mask[:, :] = 0

    elif args["inpainting"]["masktype"] == "big":
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 0
    # weight = utility.get_weight_matrix(mask, device)
    # weight = weight.to(device).float()
    weight = createWeightedMask(mask.cpu().numpy())
    weight = torch.from_numpy(weight).to(device).float()
    mask = mask.repeat(bs, 3, 1, 1)
    weight = weight.repeat(bs, 3, 1, 1)
    return mask, weight


def get_mask(img_size, bs, typ='center'):
    mask = torch.zeros(img_size, img_size).to(device).float()
    if typ == 'center':
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 1

    elif typ == 'eye':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1

    elif typ == 'face':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 1

    elif typ == "all":
        mask[:, :] = 1

    elif typ == "big":
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 1

    mask = mask.repeat(bs, 1, 1, 1)

    return mask


def createWeightedMask(mask, nsize=7):
    """Takes binary weighted mask to create weighted mask as described in
    paper.
    Arguments:
        mask - binary mask input. numpy float32 array
        nsize - pixel neighbourhood size. default = 7
    """
    ker = np.ones((nsize, nsize), dtype=np.float32)
    ker = ker / np.sum(ker)
    wmask = mask * convolve2d(mask, ker, mode='same', boundary='symm')
    return wmask


def get_model(model_name, classes):
    if model_name == 'vgg16':
        net = VGG16(num_classes=classes)
    elif model_name == 'alexnet':
        net = AlexNet(vis=True, num_classes=classes)

    return net


def calc_psnr(img1, img2):
    bs, c, h, w = img1.size()
    ten = torch.tensor(10).float().cuda()
    mse = (img1 - img2) ** 2
    # [bs, c, h, w]
    mse = torch.sum(mse, dim=1)
    mse = torch.sum(mse, dim=1)
    mse = torch.sum(mse, dim=1).view(-1, 1) / (c * h * w)
    maxI = torch.ones(bs, 1).cuda()
    psnr = 20 * torch.log(maxI / torch.sqrt(mse)) / torch.log(ten)
    return float(torch.mean(psnr))


def calc_acc(net, img, iden):
    # img = (img - 0.5) / 0.5
    img = low2high112(img)
    __, ___, out_iden = net(img)
    iden = iden.view(-1, 1)
    bs = iden.size(0)
    acc = torch.sum(iden == out_iden).item() * 1.0 / bs
    return acc


def calc_center(feat, iden):
    center = torch.from_numpy(np.load("feature/center.npy")).cuda()
    bs = feat.size(0)
    true_feat = torch.zeros(feat.size()).cuda()
    for i in range(bs):
        real_iden = iden[i].item()
        true_feat[i, :] = center[real_iden, :]
    dist = torch.sum((feat - true_feat) ** 2) / bs
    return dist.item()


def calc_knn(feat, iden):
    feats = torch.from_numpy(np.load("feature/feat.npy")).cuda()
    info = torch.from_numpy(np.load("feature/info.npy")).view(-1).long().cuda()
    bs = feat.size(0)
    tot = feats.size(0)
    knn_dist = 0
    for i in range(bs):
        knn = 1e8
        for j in range(tot):
            if info[j] == iden[i]:
                dist = torch.sum((feat[i, :] - feats[j, :]) ** 2)
                knn = min(knn, dist)
        knn_dist += knn
    return knn_dist / bs


class WarmupConstantSchedule(LambdaLR):
    """Linear warmup and then constant.
    Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
    Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )
