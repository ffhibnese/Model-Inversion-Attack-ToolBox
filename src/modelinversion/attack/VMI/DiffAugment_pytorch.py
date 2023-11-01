# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738.pdf

import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        assert x.max() <= 1 and x.min() >= 0
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=(1, 8)):
    shift_x, shift_y = x.size(2) * ratio[0] // ratio[1], x.size(3) * ratio[0] // ratio[1]
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=(1, 2)):
    cutout_size = x.size(2) * ratio[0] // ratio[1], x.size(3) * ratio[0] // ratio[1]
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'translation16': [lambda x: rand_translation(x, ratio=(1,16))],
    'translation12': [lambda x: rand_translation(x, ratio=(1,12))],
    'cutout': [rand_cutout],
    'cutout4': [lambda x: rand_cutout(x, ratio=(1,4))],
}

if __name__ == '__main__':
    import os
    import ipdb
    import utils
    import data
    import torchvision.utils as vutils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic_ckpt_path = f'{os.environ["ROOT1"]}/mm/train-ic4-cifar10/cifar10-1e-3-small-50000/discriminator.pt'
    # Data
    data_args = utils.load_args(os.path.join(os.path.dirname(ic_ckpt_path), 'args.json'))
    data_args.device = device
    dat = data.load_data(data_args.dataset, data_args.dataroot, 
                            device=device, imgsize=data_args.imageSize, Ntrain=data_args.Ntrain, Ntest=data_args.Ntest,dataset_size=data_args.dataset_size)
    reals = dat['X_train'][:100]
    # Augment
    policy = 'color,translation,cutout'
    # colored = DiffAugment(reals, policy='color')
    # translated = DiffAugment(reals, policy='translation')
    # cutout = DiffAugment(reals, policy='cutout')
    # alled = DiffAugment(reals, policy='color,translation,cutout')
    alled2 = DiffAugment(reals / 2 + .5, policy='color,translation,cutout')
    # vutils.save_image(reals / 2 + .5, 'test_augment_reals.jpeg',  nrow=10) 
    # vutils.save_image(colored / 2 + .5, 'test_augment_colored.jpeg',  nrow=10) 
    # vutils.save_image(translated / 2 + .5, 'test_augment_translated.jpeg',  nrow=10) 
    # vutils.save_image(cutout / 2 + .5, 'test_augment_cutout.jpeg',  nrow=10) 
    # vutils.save_image(alled / 2 + .5, 'test_augment_alled.jpeg',  nrow=10) 
    # vutils.save_image(alled2 , 'test_augment_alled2.jpeg',  nrow=10) 
    ipdb.set_trace()


