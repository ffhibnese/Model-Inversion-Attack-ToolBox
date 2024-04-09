import argparse
from dataloader import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm, trange
import torch.nn.functional as F
import os
import os.path as osp
import shutil
import random

# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils import *
from models import InversionNet
from facenet import FaceNet152, FaceNet
from discri import DLWGAN, DGWGAN
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from functools import partial
from tgt_models.resnet152 import ResNet152
from tgt_models.vgg16 import VGG16
from tgt_models.vit import CONFIGS, VisionTransformer
import re


parser = argparse.ArgumentParser()
parser.add_argument(
    '--name',
    type=str,
    default='vgg16',
    choices=['resnet152', 'vgg16', 'ir50', 'ViT-B_16'],
    help='type of model to use',
)
parser.add_argument(
    '--target',
    type=str,
    default='init',
    choices=['pii', 'pii-bb', 'gmi', 'ini-wb', 'ini-bb', 'full', 'full-bb', 'abl'],
    help='optimization objective',
)
parser.add_argument(
    '--mask',
    type=str,
    default='center',
    choices=['center', 'face'],
    help='type of mask',
)
parser.add_argument('--debug', action='store_true', help='print internal values')
parser.add_argument(
    '--save',
    action='store_true',
    help='save the filled images and generate virtual dataset',
)
parser.add_argument(
    '--resume',
    action='store_true',
    help='resume a previous run, which will not clear the data already generated',
)
parser.add_argument(
    '--run_file', type=str, help='only required when resume is set to true'
)

parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
parser.add_argument('--cutout_size', type=int, default=8, help='cutout size')
parser.add_argument('--num', type=int, default=200, help='number of init iterations')
parser.add_argument('--m', type=int, default=1, help='number of trys')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument(
    '--max_iter', type=int, default=1500, help='number of iterations for optimization'
)
parser.add_argument(
    '--root', type=str, default='./data/img_align_celeba/', help='path of the dataset'
)
parser.add_argument(
    '--identity_file',
    type=str,
    default='./data/identity_CelebA.txt',
    help='path of the identity file',
)
parser.add_argument(
    '--latent_dim', type=int, default=100, help='latent space dimension'
)
parser.add_argument(
    '--img_size', type=int, default=64, help='size of each image dimension'
)
parser.add_argument(
    '--patch_size', type=int, default=32, help='size of patch for local discriminator'
)
parser.add_argument('--mask_size', type=int, default=32, help='size of random mask')
opt = parser.parse_args()
print(opt)


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

priset = CelebA(split='pri')
pri_loader = DataLoader(
    priset,
    opt.batch_size,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)


num_classes = 1000

if opt.target == 'ini-bb' or opt.target == 'full-bb' or opt.target == 'pii-bb':
    root_path = osp.join('result', f'blackbox_{opt.mask}', 'model')
else:
    root_path = osp.join('result', opt.name + '_' + opt.mask, 'model')
path_N = osp.join(root_path, "Inversion_xxx.tar")
path_DG = osp.join(root_path, "DG_xxx.tar")
path_DL = osp.join(root_path, "DL_xxx.tar")
path_V = osp.join('premodels', f'{opt.name}-pri.tar')
path_I = osp.join('premodels', f'ir152-pri.tar')

ckp_N = torch.load(path_N)
ckp_DG = torch.load(path_DG)
ckp_DL = torch.load(path_DL)
ckp_V = torch.load(path_V)
ckp_I = torch.load(path_I)

acc_V, acc_I = ckp_V['acc'], ckp_I['acc']
print(f'acc_V: {acc_V}, acc_I: {acc_I}')

Net = InversionNet()
DL = DLWGAN()
DG = DGWGAN()
if opt.name == 'resnet152':
    V = ResNet152(num_classes=num_classes)
elif opt.name == 'vgg16':
    V = VGG16(num_classes=num_classes)
elif opt.name == 'ir50':
    V = FaceNet(num_classes=num_classes)
elif opt.name == 'ViT-B_16':
    V = VisionTransformer(CONFIGS[opt.name], num_classes=num_classes)
I = FaceNet152(num_classes=num_classes)

Net.load_state_dict(ckp_N['state_dict'])
DL.load_state_dict(ckp_DL['state_dict'])
DG.load_state_dict(ckp_DG['state_dict'])
V.load_state_dict(ckp_V['state_dict'])
I.load_state_dict(ckp_I['state_dict'])

# Net = torch.nn.DataParallel(Net)
# DL = torch.nn.DataParallel(DL)
# DG = torch.nn.DataParallel(DG)
# V = torch.nn.DataParallel(V)
# I = torch.nn.DataParallel(I)

Net = Net.to(device)
DL = DL.to(device)
DG = DG.to(device)
V = V.to(device)
I = I.to(device)

DG.eval()
DL.eval()
Net.eval()
V.eval()
I.eval()

if opt.name == 'ir50':
    scale_fnV = low2high112
elif opt.name == 'ViT-B_16':
    scale_fnV = low2high224
else:
    scale_fnV = None
scale_fnI = low2high112

mpv = np.array([0.5189, 0.4346, 0.3886])
mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).cuda()

for param in Net.parameters():
    param.requires_grad = False
for param in DG.parameters():
    param.requires_grad = False
for param in DL.parameters():
    param.requires_grad = False
for param in V.parameters():
    param.requires_grad = False
for param in I.parameters():
    param.requires_grad = False


tag_run = f'{opt.name}_{opt.target}_{opt.mask}'
if opt.save:
    out_dir = osp.join('data', tag_run)
    img_dir = osp.join(out_dir, 'img')
    file_name = 'identity.txt'
    file_path = osp.join(out_dir, file_name)

    if not opt.resume:
        if osp.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        os.mkdir(img_dir)
    else:
        assert osp.isdir(out_dir)
        assert osp.isdir(img_dir)
        assert osp.exists(file_path)

        with open(opt.run_file) as f:
            lines = f.readlines()

        fmt = r'top1_V:\s(.+?),\stop5_V:\s(.+?),\stop1_I:\s(.+?),\stop5_I:\s(.+?),\s\spsnr:\s(.+?):.+\|\s(.+?)/(.+?)\s'
        for line in reversed(lines):
            result = re.search(fmt, line)
            if result is not None:
                group = result.groups()
                top1I_pre = float(group[2])
                top5I_pre = float(group[3])
                top1V_pre = float(group[0])
                top5V_pre = float(group[1])
                psnr_pre = float(group[4])
                start_idx = int(group[5])
                break

        # with open(file_path, 'r') as f:
        #     lines = f.readlines()
        #     assert len(lines) % opt.batch_size == 0
        #     start_idx = len(lines) // opt.batch_size


print('finished loading model and data...cheers!')


easy_cutout_area = partial(
    gen_cutout_area,
    cutout_size=opt.cutout_size,
    img_size=opt.img_size,
    patch_size=opt.patch_size,
)
easy_cutout = partial(cutout, mpv=mpv, img_size=opt.img_size, bs=opt.batch_size)


@torch.no_grad()
def calc_Itop1(rec_data, gt_label, scale_fnI=scale_fnI):
    if scale_fnI is not None:
        high_rec_data = scale_fnI(rec_data)
    else:
        high_rec_data = rec_data

    _, pred_outputs, _ = I(high_rec_data)
    conf, _ = torch.max(pred_outputs, dim=1)
    conf = float(torch.mean(conf))

    pred_class_idx = torch.argmax(pred_outputs, dim=1)
    correct_top1 = pred_class_idx[pred_class_idx == gt_label].shape[0]
    correct_top1 /= rec_data.shape[0]

    pred_class_top5 = torch.topk(pred_outputs, k=5, dim=-1).indices
    gt_class_idx = gt_label.unsqueeze(-1).repeat(1, 5)
    correct_top5 = int(
        torch.sum((torch.sum((gt_class_idx == pred_class_top5), dim=1) > 0), dim=0)
    )
    correct_top5 /= rec_data.shape[0]

    return correct_top1, correct_top5, conf


@torch.no_grad()
def calc_Vtop1(rec_data, gt_label, scale_fnV=scale_fnV):
    if scale_fnV is not None:
        high_rec_data = scale_fnV(rec_data)
    else:
        high_rec_data = rec_data

    _, pred_outputs, _ = V(high_rec_data)
    conf, _ = torch.max(pred_outputs, dim=1)
    conf = float(torch.mean(conf))

    pred_class_idx = torch.argmax(pred_outputs, dim=1)
    correct_top1 = pred_class_idx[pred_class_idx == gt_label].shape[0]
    correct_top1 /= rec_data.shape[0]

    pred_class_top5 = torch.topk(pred_outputs, k=5, dim=-1).indices
    gt_class_idx = gt_label.unsqueeze(-1).repeat(1, 5)
    correct_top5 = int(
        torch.sum((torch.sum((gt_class_idx == pred_class_top5), dim=1) > 0), dim=0)
    )
    correct_top5 /= rec_data.shape[0]

    return correct_top1, correct_top5, conf


class FixedCutOut(nn.Module):
    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, img):
        out = easy_cutout(img, self.area)
        return out


def define_trans():
    trans = []
    for i in range(4):
        for j in range(4):
            area = ((16 * i, 16 * j), (16, 16))
            trans.append(FixedCutOut(area))

    return trans


def define_trans_plus(m):
    num_split = math.sqrt(m)
    assert num_split.is_integer()
    num_split = int(num_split)
    patch_size = 64 // num_split

    trans = []
    for i in range(num_split):
        for j in range(num_split):
            area = ((patch_size * i, patch_size * j), (patch_size, patch_size))
            trans.append(FixedCutOut(area))

    return trans


def wrapper_V(imgs):
    if scale_fnV is not None:
        imgs = scale_fnV(imgs)
    feat, out, iden = V(imgs)
    return feat, out, iden


@torch.no_grad()
def init_latent(imgs, mask, num):
    trans = define_trans()

    imgs_masked = imgs - imgs * mask + mpv * mask
    inp = torch.cat((imgs_masked, mask), dim=1)

    probs_avg = torch.zeros((opt.batch_size, 1000)).to(device)
    for i in trange(num):
        z = torch.randn(opt.batch_size, 100).to(device)
        z = torch.clamp(z, -1, 1)

        filled_imgs = Net((inp, z))
        filled_imgs = imgs - imgs * mask + filled_imgs * mask

        _, probs, _ = wrapper_V(filled_imgs)
        probs_avg += probs
        for tr in trans:
            _, probs_tr, _ = wrapper_V(tr(filled_imgs))
            probs_avg += probs_tr

    probs_avg /= (1 + len(trans)) * num
    pseudo_labels = torch.argmax(probs_avg, dim=1)

    return pseudo_labels


@torch.no_grad()
def init_latent_blackbox(imgs, mask, gt_labels, num):
    '''
    Returns:
        z_bank (torch.Tensor): shape (B * dim_z) tensor
        id_bank (torch.Tensor): shape (B) tensor
    '''
    trans = define_trans()

    imgs_masked = imgs - imgs * mask + mpv * mask
    inp = torch.cat((imgs_masked, mask), dim=1)

    probs_avg = torch.zeros((opt.batch_size, 1000)).to(device)
    z_dict = None  # B * num * dim_z
    conf_dict = None  # B * num
    label_dict = None  # B * num
    z_bank = None  # B * dim_z
    for i in trange(num):
        z = torch.randn(opt.batch_size, 100).to(device)
        z = torch.clamp(z, -1, 1)
        z_dict = (
            torch.cat((z_dict, z.unsqueeze(1)), dim=1)
            if z_dict is not None
            else z.unsqueeze(1)
        )
        filled_imgs = Net((inp, z))
        filled_imgs = imgs - imgs * mask + filled_imgs * mask

        _, probs, labels = wrapper_V(filled_imgs)
        if gt_labels is None:
            probs_avg += probs
            for tr in trans:
                _, conf_tr, _ = wrapper_V(tr(filled_imgs))
                probs_avg += conf_tr

        label_dict = (
            torch.cat((label_dict, labels.unsqueeze(-1)), dim=1)
            if label_dict is not None
            else labels.unsqueeze(-1)
        )
        conf, _ = torch.max(probs, dim=1, keepdim=True)
        conf_dict = (
            torch.cat((conf_dict, conf), dim=1) if conf_dict is not None else conf
        )

    probs_avg /= (1 + len(trans)) * num

    if gt_labels is None:
        id_bank = torch.zeros(opt.batch_size, dtype=int).to(device)  # B
        id_bank_all = torch.argsort(probs_avg, dim=1, descending=True).to(device)

        # check whether some ids in id_bank does not belong to id(img_trans)
        # i: batch
        for i, idens in enumerate(id_bank_all):
            for iden in idens:
                if len(label_dict[i][label_dict[i] == iden]) != 0:
                    id_bank[i] = iden
                    break
    else:
        id_bank = gt_labels

    for i, iden in enumerate(id_bank):
        z_all = z_dict[i]
        conf_all = conf_dict[i]

        z_potential = z_all[label_dict[i] == iden]
        conf_potential = conf_all[label_dict[i] == iden]

        if z_potential.shape[0] > 0:
            z_select = z_potential[torch.argmax(conf_potential)].unsqueeze(0)
        else:
            z_select = torch.randn(1, 100).to(device)

        z_bank = (
            torch.cat((z_bank, z_select), dim=0) if z_bank is not None else z_select
        )

    return z_bank, id_bank


@torch.no_grad()
def init_full(imgs, mask, gt_labels, num, k=5):
    '''
    Returns:
        z_bank (torch.Tensor): shape (B * k * dim_z) tensor
        id_bank (torch.Tensor): shape (B * k) tensor
    '''
    trans = define_trans_plus(16)

    imgs_masked = imgs - imgs * mask + mpv * mask
    inp = torch.cat((imgs_masked, mask), dim=1)

    probs_avg = torch.zeros((opt.batch_size, 1000)).to(device)
    z_dict = None  # B * num * dim_z
    conf_dict = None  # B * num
    label_dict = None  # B * num
    z_bank = None  # B * k * dim_z
    for i in trange(num):
        z = torch.randn(opt.batch_size, 100).to(device)
        z = torch.clamp(z, -1, 1)
        z_dict = (
            torch.cat((z_dict, z.unsqueeze(1)), dim=1)
            if z_dict is not None
            else z.unsqueeze(1)
        )

        filled_imgs = Net((inp, z))
        filled_imgs = imgs - imgs * mask + filled_imgs * mask

        _, probs, labels = wrapper_V(filled_imgs)
        if gt_labels is None:
            probs_avg += probs
            for tr in trans:
                _, conf_tr, _ = wrapper_V(tr(filled_imgs))
                probs_avg += conf_tr

        _, probs, labels = wrapper_V(filled_imgs)
        label_dict = (
            torch.cat((label_dict, labels.unsqueeze(-1)), dim=1)
            if label_dict is not None
            else labels.unsqueeze(-1)
        )
        conf, _ = torch.max(probs, dim=1, keepdim=True)
        conf_dict = (
            torch.cat((conf_dict, conf), dim=1) if conf_dict is not None else conf
        )

    probs_avg /= (1 + len(trans)) * num

    if gt_labels is None:
        id_bank = torch.zeros(opt.batch_size, dtype=int).to(device)  # B
        id_bank_all = torch.argsort(probs_avg, dim=1, descending=True).to(device)

        # check whether some ids in id_bank does not belong to id(img_trans)
        # i: batch
        for i, idens in enumerate(id_bank_all):
            for iden in idens:
                if len(label_dict[i][label_dict[i] == iden]) != 0:
                    id_bank[i] = iden
                    break
    else:
        id_bank = gt_labels

    for i, iden in enumerate(id_bank):
        z_all = z_dict[i]
        conf_all = conf_dict[i]
        z_selects = None

        z_potential = z_all[label_dict[i] == iden]
        conf_potential = conf_all[label_dict[i] == iden]

        if z_potential.shape[0] >= k:
            z_selects = z_potential[torch.topk(conf_potential, k=k).indices]
        else:
            z_selects = z_potential[
                torch.topk(conf_potential, k=z_potential.shape[0]).indices
            ]
            z_remains = torch.randn(k - z_potential.shape[0], 100).to(device)
            z_selects = torch.cat((z_selects, z_remains), dim=0)

        z_selects = z_selects.unsqueeze(0)
        z_bank = (
            torch.cat((z_bank, z_selects), dim=0) if z_bank is not None else z_selects
        )

    return z_bank, id_bank


@torch.no_grad()
def init_naive(imgs, mask, gt_labels, num, k=5):
    '''
    Returns:
        z_bank (torch.Tensor): shape (B * k * dim_z) tensor
        id_bank (torch.Tensor): shape (B * k) tensor
    '''

    imgs_masked = imgs - imgs * mask + mpv * mask
    inp = torch.cat((imgs_masked, mask), dim=1)

    probs_avg = torch.zeros((opt.batch_size, 1000)).to(device)
    z_dict = None  # B * num * dim_z
    conf_dict = None  # B * num
    label_dict = None  # B * num
    z_bank = None  # B * k * dim_z
    for i in trange(num):
        z = torch.randn(opt.batch_size, 100).to(device)
        z = torch.clamp(z, -1, 1)
        z_dict = (
            torch.cat((z_dict, z.unsqueeze(1)), dim=1)
            if z_dict is not None
            else z.unsqueeze(1)
        )

        filled_imgs = Net((inp, z))
        filled_imgs = imgs - imgs * mask + filled_imgs * mask

        _, probs, labels = wrapper_V(filled_imgs)
        if gt_labels is None:
            probs_avg += probs

        _, probs, labels = wrapper_V(filled_imgs)
        label_dict = (
            torch.cat((label_dict, labels.unsqueeze(-1)), dim=1)
            if label_dict is not None
            else labels.unsqueeze(-1)
        )
        conf, _ = torch.max(probs, dim=1, keepdim=True)
        conf_dict = (
            torch.cat((conf_dict, conf), dim=1) if conf_dict is not None else conf
        )

    probs_avg /= num

    if gt_labels is None:
        id_bank = torch.zeros(opt.batch_size, dtype=int).to(device)  # B
        id_bank_all = torch.argsort(probs_avg, dim=1, descending=True).to(device)

        # check whether some ids in id_bank does not belong to id(img_trans)
        # i: batch
        for i, idens in enumerate(id_bank_all):
            for iden in idens:
                if len(label_dict[i][label_dict[i] == iden]) != 0:
                    id_bank[i] = iden
                    break
    else:
        id_bank = gt_labels

    for i, iden in enumerate(id_bank):
        z_all = z_dict[i]
        conf_all = conf_dict[i]
        z_selects = None

        z_potential = z_all[label_dict[i] == iden]
        conf_potential = conf_all[label_dict[i] == iden]

        if z_potential.shape[0] >= k:
            z_selects = z_potential[torch.topk(conf_potential, k=k).indices]
        else:
            z_selects = z_potential[
                torch.topk(conf_potential, k=z_potential.shape[0]).indices
            ]
            z_remains = torch.randn(k - z_potential.shape[0], 100).to(device)
            z_selects = torch.cat((z_selects, z_remains), dim=0)

        z_selects = z_selects.unsqueeze(0)
        z_bank = (
            torch.cat((z_bank, z_selects), dim=0) if z_bank is not None else z_selects
        )

    return z_bank, id_bank


def attack(imgs: torch.tensor, mask: torch.tensor, gt_labels: torch.tensor):
    imgs_masked = imgs - imgs * mask + mpv * mask
    inp = torch.cat((imgs_masked, mask), dim=1)

    if opt.target == 'pii' or opt.target == 'pii-bb':
        z = torch.randn(opt.batch_size, 100).to(device)
        z.requires_grad = True
        v = torch.zeros(opt.batch_size, 100).to(device)
        momentum = 0.9

        bar = tqdm(range(opt.max_iter))
        for i in bar:
            filled_imgs = Net((inp, z))
            filled_imgs = imgs - imgs * mask + filled_imgs * mask

            hole_area = gen_hole_area(
                (opt.patch_size, opt.patch_size), (opt.img_size, opt.img_size)
            )
            filled_img_patchs = crop(filled_imgs, hole_area)
            _, logit_dl = DL(filled_img_patchs)
            _, logit_dg = DG(filled_imgs)

            loss = -torch.mean(logit_dl) - torch.mean(logit_dg)
            if z.grad is not None:
                z.grad.data.zero_()
            loss.backward()
            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - 0.01 * gradient
            z = z + (-momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -1, 1)
            z.requires_grad = True
            bar.set_description('L: {:.3f}'.format(loss))

    elif opt.target == 'gmi' or opt.target == 'ini-wb' or opt.target == 'full':
        if opt.target == 'gmi':
            target_labels = gt_labels
            z_bank = torch.randn(opt.batch_size, opt.m, 100).to(device)
        elif opt.target == 'ini-wb':
            z_bank, target_labels = init_full(imgs, mask, gt_labels=None, num=opt.num)
        elif opt.target == 'full':
            target_labels = gt_labels
            z_bank, _ = init_full(imgs, mask, gt_labels=gt_labels, num=opt.num)

        z_best = torch.zeros(opt.batch_size, 100).to(device)
        min_loss_id = torch.ones(opt.batch_size).to(device) * 999

        nll_loss = nn.NLLLoss(reduction='none')

        for idx in range(opt.m):
            z = z_bank[:, idx]
            z.requires_grad = True
            v = torch.zeros(opt.batch_size, 100).to(device)
            momentum = 0.9

            # writer = SummaryWriter(log_dir=osp.join("logs", tag_run))
            bar = tqdm(range(opt.max_iter))
            for i in bar:
                filled_imgs = Net((inp, z))
                filled_imgs = imgs - imgs * mask + filled_imgs * mask

                hole_area = gen_hole_area(
                    (opt.patch_size, opt.patch_size), (opt.img_size, opt.img_size)
                )
                filled_img_patchs = crop(filled_imgs, hole_area)
                _, logit_dl = DL(filled_img_patchs)
                _, logit_dg = DG(filled_imgs)

                loss_dis = -torch.mean(logit_dl) - torch.mean(logit_dg)

                _, probs_gen, _ = wrapper_V(filled_imgs)
                loss_id = nll_loss(torch.log(probs_gen), target_labels)
                loss = loss_dis + 100 * loss_id.mean()

                # writer.add_scalar('L_dis', loss_dis, i)
                # writer.add_scalar('L_cls', loss_cls, i)

                if z.grad is not None:
                    z.grad.data.zero_()
                loss.backward()
                v_prev = v.clone()
                gradient = z.grad.data
                v = momentum * v - 0.01 * gradient
                z = z + (-momentum * v_prev + (1 + momentum) * v)
                z = torch.clamp(z.detach(), -1, 1)
                z.requires_grad = True

                bar.set_description(
                    'L: {:.3f}, L_dis: {:.3f}, L_id: {:.3f}'.format(
                        loss, loss_dis, loss_id.mean()
                    )
                )

                if opt.debug:
                    print()
                    print()
                    print(calc_Vtop1(filled_imgs, gt_labels))
                    print(calc_Itop1(filled_imgs, gt_labels))

            # writer.close()

            with torch.no_grad():
                filled_imgs = Net((inp, z))
                filled_imgs = imgs - imgs * mask + filled_imgs * mask
                _, probs_gen, _ = wrapper_V(filled_imgs)
                loss_id = nll_loss(torch.log(probs_gen), target_labels)

                z_best[loss_id < min_loss_id] = z[loss_id < min_loss_id]
                min_loss_id[loss_id < min_loss_id] = loss_id[loss_id < min_loss_id]

        z = z_best

    elif opt.target == 'ini-bb' or opt.target == 'full-bb':
        if opt.target == 'ini-bb':
            z, pred_ids = init_latent_blackbox(imgs, mask, None, opt.num)
        else:
            z, pred_ids = init_latent_blackbox(imgs, mask, gt_labels, opt.num)
        z.requires_grad = True
        v = torch.zeros(opt.batch_size, 100).to(device)
        momentum = 0.9

        bar = tqdm(range(opt.max_iter))
        for i in bar:
            filled_imgs = Net((inp, z))
            filled_imgs = imgs - imgs * mask + filled_imgs * mask

            hole_area = gen_hole_area(
                (opt.patch_size, opt.patch_size), (opt.img_size, opt.img_size)
            )
            filled_img_patchs = crop(filled_imgs, hole_area)
            _, logit_dl = DL(filled_img_patchs)
            _, logit_dg = DG(filled_imgs)

            loss = -torch.mean(logit_dl) - torch.mean(logit_dg)
            if z.grad is not None:
                z.grad.data.zero_()
            loss.backward()
            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - 0.01 * gradient
            z = z + (-momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -1, 1)
            z.requires_grad = True
            bar.set_description('L: {:.3f}'.format(loss))

    elif opt.target == 'abl':
        z_bank, target_labels = init_naive(imgs, mask, gt_labels=None, num=opt.num)

        z_best = torch.zeros(opt.batch_size, 100).to(device)
        min_loss_id = torch.ones(opt.batch_size).to(device) * 999

        nll_loss = nn.NLLLoss(reduction='none')

        for idx in range(opt.m):
            z = z_bank[:, idx]
            z.requires_grad = True
            v = torch.zeros(opt.batch_size, 100).to(device)
            momentum = 0.9

            # writer = SummaryWriter(log_dir=osp.join("logs", tag_run))
            bar = tqdm(range(opt.max_iter))
            for i in bar:
                filled_imgs = Net((inp, z))
                filled_imgs = imgs - imgs * mask + filled_imgs * mask

                hole_area = gen_hole_area(
                    (opt.patch_size, opt.patch_size), (opt.img_size, opt.img_size)
                )
                filled_img_patchs = crop(filled_imgs, hole_area)
                _, logit_dl = DL(filled_img_patchs)
                _, logit_dg = DG(filled_imgs)

                loss_dis = -torch.mean(logit_dl) - torch.mean(logit_dg)

                _, probs_gen, _ = wrapper_V(filled_imgs)
                loss_id = nll_loss(torch.log(probs_gen), target_labels)
                loss = loss_dis + 100 * loss_id.mean()

                # writer.add_scalar('L_dis', loss_dis, i)
                # writer.add_scalar('L_cls', loss_cls, i)

                if z.grad is not None:
                    z.grad.data.zero_()
                loss.backward()
                v_prev = v.clone()
                gradient = z.grad.data
                v = momentum * v - 0.01 * gradient
                z = z + (-momentum * v_prev + (1 + momentum) * v)
                z = torch.clamp(z.detach(), -1, 1)
                z.requires_grad = True

                bar.set_description(
                    'L: {:.3f}, L_dis: {:.3f}, L_id: {:.3f}'.format(
                        loss, loss_dis, loss_id.mean()
                    )
                )

                if opt.debug:
                    print()
                    print()
                    print(calc_Vtop1(filled_imgs, gt_labels))
                    print(calc_Itop1(filled_imgs, gt_labels))

            # writer.close()

            with torch.no_grad():
                filled_imgs = Net((inp, z))
                filled_imgs = imgs - imgs * mask + filled_imgs * mask
                _, probs_gen, _ = wrapper_V(filled_imgs)
                loss_id = nll_loss(torch.log(probs_gen), target_labels)

                z_best[loss_id < min_loss_id] = z[loss_id < min_loss_id]
                min_loss_id[loss_id < min_loss_id] = loss_id[loss_id < min_loss_id]

        z = z_best

    filled_imgs = Net((inp, z))
    filled_imgs = imgs - imgs * mask + filled_imgs * mask

    return filled_imgs


def run():
    if opt.resume:
        top1I_all = top1I_pre * start_idx
        top5I_all = top5I_pre * start_idx
        top1V_all = top1V_pre * start_idx
        top5V_all = top5V_pre * start_idx
        psnr_all = psnr_pre * start_idx
    else:
        top1I_all = 0
        top5I_all = 0
        top1V_all = 0
        top5V_all = 0
        psnr_all = 0

    pbar = tqdm(pri_loader)
    for idx, batch in enumerate(pbar):
        if opt.resume:
            if idx < start_idx:
                continue
        imgs, _, ids = batch
        imgs = imgs.to(device)
        ids = ids.to(device)

        mask = get_mask(opt.img_size, opt.batch_size, typ=opt.mask)
        filled_imgs = attack(imgs, mask, ids)
        top1_V, top5_V, conf_V = calc_Vtop1(filled_imgs, ids)
        top1_I, top5_I, conf_I = calc_Itop1(filled_imgs, ids)
        psnr = calc_psnr(imgs, filled_imgs)

        top1V_all += top1_V
        top5V_all += top5_V
        top1I_all += top1_I
        top5I_all += top5_I
        psnr_all += psnr

        top1V_avg = top1V_all / (idx + 1)
        top5V_avg = top5V_all / (idx + 1)
        top1I_avg = top1I_all / (idx + 1)
        top5I_avg = top5I_all / (idx + 1)
        psnr_avg = psnr_all / (idx + 1)

        pbar.set_description(
            f'top1_V: {top1V_avg:.3f}, top5_V: {top5V_avg:.3f}, top1_I: {top1I_avg:.3f}, top5_I: {top5I_avg:.3f},  psnr: {psnr_avg:.3f}'
        )

        if opt.save:
            with torch.no_grad():
                _, _, labels = wrapper_V(filled_imgs)

                f = open(file_path, 'a+')
                for i in range(opt.batch_size):
                    img_name = str(idx * opt.batch_size + i) + '.png'
                    img_path = osp.join(img_dir, img_name)
                    save_image(filled_imgs[i], img_path)
                    f.write('{} {}\n'.format(img_path, str(int(labels[i]))))

                f.close()


run()
