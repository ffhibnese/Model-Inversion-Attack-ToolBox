
import os
import time
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import kornia
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tv_trans
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

from modelinversion.metrics.base import DataLoader

from . import utils
from . import losses as L
from ..base import BaseGANTrainArgs, BaseGANTrainer
from ...models import *
from ...utils import walk_imgs, print_as_yaml
from .code.m_cgan import ResNetGenerator, SNResNetProjectionDiscriminator
import json

def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = args.results_root
    os.makedirs(root, exist_ok=True)
        # if not args.no_tensorboard:
        #     from tensorboardX import SummaryWriter
        #     writer = SummaryWriter(root)
        # else:
    writer = None

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    if args.num_classes > args.n_eval_batches:
        args.n_eval_batches = args.num_classes
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size // 4

    if args.calc_FID:
        args.n_fid_batches = args.n_fid_images // args.batch_size

    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, writer

def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)

    return real, y


def sample_from_gen(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    pseudo_y = utils.sample_pseudo_labels(
        num_classes, args.batch_size, device
    )

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z
         
@dataclass
class PlgmiGANTrainArgs(BaseGANTrainArgs):
    top_n: int = 30
    target_name: str = 'vgg16'
    target_dataset_name: str = 'celeba'
    # num_classes: int = 1000
    augment: Callable = field(default_factory=lambda: kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    ))
    
    coef_inv_loss: float = 0.2
    lr: float= 0.0002
    # beta1: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.9
    z_dim = 200
    gen_distribution = 'normal'
            
class PlgmiGANTrainer(BaseGANTrainer):
    
    def __init__(self, args: PlgmiGANTrainArgs, folder_manager, more_args, **kwargs) -> None:
        for k, v in args.__dict__.items():
            setattr(more_args, k, v)
        more_args.num_classes = NUM_CLASSES[args.target_dataset_name]
        super().__init__(more_args, folder_manager, **kwargs)
        self.args: PlgmiGANTrainArgs
        self.num_classes = NUM_CLASSES[args.target_dataset_name]
        
        
        
        self.src_dataset_dir = more_args.data_root # os.path.join(folder_manager.config.dataset_dir, args.dataset_name, 'split', 'public')
        self.path_T = more_args.target_ckpt_path # '/data/yuhongyao/Model_Inversion_Attack_ToolBox/checkpoints/target_eval/hdceleba/resnet152_celeba.pt'
        
        self.dst_dataset_dir =  os.path.join(folder_manager.config.cache_dir, args.dataset_name, args.target_name)
        
    def get_tag(self) -> str:
        args = self.args
        return f'plgmi_high_{args.top_n}_{args.dataset_name}_{args.target_name}_{args.target_dataset_name}'
    
    def get_method_name(self) -> str:
        return 'PLGMI_high'
    
    def _check_select_topn(self):
        ret = True
        for i in range(self.num_classes):
            dirname = os.path.join(self.dst_dataset_dir, f'{i}')
            if not os.path.exists(dirname):
                ret = False
            elif len(os.listdir(dirname)) != self.args.top_n:
                os.system(f'rm -rf {dirname}')
                ret = False
            
        return ret
    
    def get_trainloader(self) -> DataLoader:
        dataset = ImageFolder(self.dst_dataset_dir, transform=tv_trans.Compose([
                # Image.open,
                tv_trans.ToTensor(), 
                # tv_trans.CenterCrop((800,800)),
                # tv_trans.Resize((256, 256), antialias=True)
            ]))
        dataloader = DataLoader(dataset, self.args.batch_size, shuffle=True)
        return dataloader
        
    def prepare_training(self):
        # return "maomao"
        args = self.args
        self.G = ResNetGenerator(num_classes=self.num_classes, distribution=args.gen_distribution)
        self.D = SNResNetProjectionDiscriminator(num_classes=self.num_classes)
        # self.G.load_state_dict(torch.load('checkpoints/PLGMI_high/plgmi_high_metfaces_resnet18_facescrub_G.pt')['state_dict'])
        # self.D.load_state_dict(torch.load('checkpoints/PLGMI_high/plgmi_high_metfaces_resnet18_facescrub_D.pt')['state_dict'])
        self.G = nn.DataParallel(self.G).to(args.device)
        self.D = nn.DataParallel(self.D).to(args.device)
        self.T = get_model(args.target_name, args.target_dataset_name, device=args.device, backbone_pretrain=False, defense_type=args.defense_type)
        # self.folder_manager.load_target_model_state_dict(self.T, args.target_dataset_name, args.target_name, device=args.device, defense_type=args.defense_type)
        self.T.load_state_dict(torch.load(self.path_T, map_location=args.device)['state_dict'])
        self.T.eval()
        
        # self.folder_manager.load_state_dict(self.G, [self.method_name, f'{self.tag}_G.pt'], self.args.device, self.args.defense_type)
        # self.folder_manager.load_state_dict(self.D, [self.method_name, f'{self.tag}_D.pt'], self.args.device, self.args.defense_type)
        
        self.optim_G = torch.optim.Adam(self.G.parameters(), args.lr, (args.beta1, args.beta2))
        self.optim_D = torch.optim.Adam(self.D.parameters(), args.lr, (args.beta1, args.beta2))
        
        if not self._check_select_topn():
            print(f'start top n selection from {self.src_dataset_dir} to {self.dst_dataset_dir}')
            src_img_paths = walk_imgs(self.src_dataset_dir)
            
            trans = tv_trans.Compose([
                Image.open,
                tv_trans.ToTensor(), 
                # tv_trans.CenterCrop((800,800)),
                # tv_trans.Resize((256, 256), antialias=True)
            ])
            
            with torch.no_grad():
                # src_imgs = [trans(p) for p in tqdm(src_img_paths)]
                # src_imgs = torch.stack(src_imgs, dim=0)
                src_scores = []
                total_num = len(src_img_paths)
                for i in tqdm(range((total_num-1) // args.batch_size + 1)):
                    start_idx = i * args.batch_size
                    end_idx = min(start_idx + args.batch_size, total_num)
                    # batch_paths = src_img_paths[start_idx:end_idx]
                    use_paths = src_img_paths[start_idx:end_idx]
                    batch_imgs = torch.stack([trans(p) for p in use_paths], dim=0).to(args.device)
                    # batch_imgs = src_imgs[start_idx:end_idx].to(args.device)
                    batch_scores = self.T(batch_imgs).result.softmax(dim=-1).cpu()
                    src_scores.append(batch_scores)
                src_scores = torch.cat(src_scores, dim=0)
                
                i = 0
                for label in tqdm(range(self.num_classes)):
                    dst_dir = os.path.join(self.dst_dataset_dir, f'{label}')
                    os.makedirs(dst_dir, exist_ok=True)
                    scores = src_scores[:, label]
                    _, indice = torch.topk(scores, k=args.top_n)
                    # torch.save(src_imgs, os.path.join(dst_dir, f'{label}.pt'))
                    indice = indice.numpy().tolist()
                    for idx in indice:
                        # torch
                        os.system(f'cp {src_img_paths[idx]} {dst_dir}/')
                        
        def _noise_adder(img):
            return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img    
        ds = ImageFolder(self.dst_dataset_dir, tv_trans.Compose([
            tv_trans.ToTensor(),
            _noise_adder
        ]))
        
        def InfiniteSampler(n):
            # i = 0
            i = n - 1
            order = np.random.permutation(n)
            while True:
                yield order[i]
                i += 1
                if i >= n:
                    np.random.seed()
                    order = np.random.permutation(n)
                    i = 0
        
        class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
            def __init__(self, data_source):
                self.num_samples = len(data_source)

            def __iter__(self):
                return iter(InfiniteSampler(self.num_samples))

            def __len__(self):
                return 2 ** 31
        
        from torch.utils.data import DataLoader
        # data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        data_loader = iter(DataLoader(
            ds, 32,
            sampler=InfiniteSamplerWrapper(ds),
        ))
        
        dis = self.D
        gen = self.G
        target_model = self.T
        
        opt_gen = torch.optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
        opt_dis = torch.optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))
        # get adversarial loss
        gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
        dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)
        
        args, writer = prepare_results_dir(args)
        print(' Initialized models...\n')

        if args.args_path is not None:
            print(' Load weights...\n')
            prev_args, gen, opt_gen, dis, opt_dis = utils.resume_from_args(
                args.args_path, args.gen_ckpt_path, args.dis_ckpt_path
            )
        # data augmentation module in stage-1 for the generated images
        aug_list = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        )

        # Training loop
        # from tqdm import tqdm
        for n_iter in tqdm(range(1, args.max_iteration + 1)):
            # ==================== Beginning of 1 iteration. ====================
            _l_g = .0
            cumulative_inv_loss = 0.
            cumulative_loss_dis = .0

            cumulative_target_acc = .0
            target_correct = 0
            count = 0
            for i in range(args.n_dis):  # args.ndis=5, Gen update 1 time, Dis update ndis times.
                if i == 0:
                    fake, pseudo_y, _ = sample_from_gen(args, args.device, args.num_classes, gen)
                    dis_fake = dis(fake, pseudo_y)
                    # random transformation on the generated images
                    fake_aug = aug_list(fake)
                    # calc the L_inv
                    if args.inv_loss_type == 'ce':
                        inv_loss = L.cross_entropy_loss(target_model(fake_aug).result, pseudo_y)
                    elif args.inv_loss_type == 'margin':
                        inv_loss = L.max_margin_loss(target_model(fake_aug).result, pseudo_y)
                    elif args.inv_loss_type == 'poincare':
                        inv_loss = L.poincare_loss(target_model(fake_aug).result, pseudo_y)
                    # not used
                    if args.relativistic_loss:
                        real, y = sample_from_data(args, args.device, data_loader)
                        dis_real = dis(real, y)
                    else:
                        dis_real = None
                    # calc the loss of G
                    loss_gen = gen_criterion(dis_fake, dis_real)
                    loss_all = loss_gen + inv_loss * args.alpha
                    # update the G
                    gen.zero_grad()
                    loss_all.backward()
                    opt_gen.step()
                    _l_g += loss_gen.item()

                    cumulative_inv_loss += inv_loss.item()

                    # if n_iter % 10 == 0 and writer is not None:
                    #     writer.add_scalar('gen', _l_g, n_iter)
                    #     writer.add_scalar('inv', cumulative_inv_loss, n_iter)
                # generate fake images
                fake, pseudo_y, _ = sample_from_gen(args, args.device, args.num_classes, gen)
                # sample the real images
                real, y = sample_from_data(args, args.device, data_loader)
                # calc the loss of D
                dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
                loss_dis = dis_criterion(dis_fake, dis_real)
                # update D
                dis.zero_grad()
                loss_dis.backward()
                opt_dis.step()

                cumulative_loss_dis += loss_dis.item()

                with torch.no_grad():
                    count += fake.shape[0]
                    T_logits = target_model(fake).result
                    T_preds = T_logits.max(1, keepdim=True)[1]
                    target_correct += T_preds.eq(pseudo_y.view_as(T_preds)).sum().item()
                    cumulative_target_acc += round(target_correct / count, 4)

                # if n_iter % 10 == 0 and i == args.n_dis - 1 and writer is not None:
                #     cumulative_loss_dis /= args.n_dis
                #     cumulative_target_acc /= args.n_dis
                #     writer.add_scalar('dis', cumulative_loss_dis, n_iter)
                #     writer.add_scalar('target acc', cumulative_target_acc, n_iter)
            # ==================== End of 1 iteration. ====================

            if n_iter % args.log_interval == 0:
                print(
                    'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                        n_iter, args.max_iteration, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                        cumulative_target_acc, ))
                # if not args.no_image:
                #     writer.add_image(
                #         'fake', torchvision.utils.make_grid(
                #             fake, nrow=4, normalize=True, scale_each=True))
                #     writer.add_image(
                #         'real', torchvision.utils.make_grid(
                #             real, nrow=4, normalize=True, scale_each=True))
                # Save previews
                utils.save_images(
                    n_iter, n_iter // args.checkpoint_interval, args.results_root,
                    args.train_image_root, fake, real
                )
            if n_iter % args.checkpoint_interval == 0:
                # Save checkpoints!
                # utils.save_checkpoints(
                #     args, n_iter, n_iter // args.checkpoint_interval,
                #     gen.module, opt_gen, dis.module, opt_dis
                # )
                torch.save({'state_dict': gen.module.state_dict()}, os.path.join(self.folder_manager.config.cache_dir, 'G.pth'))
                torch.save({'state_dict': dis.module.state_dict()}, os.path.join(self.folder_manager.config.cache_dir, 'D.pth'))
        torch.save({'state_dict': gen.module.state_dict()}, os.path.join(self.folder_manager.config.cache_dir, 'G.pth'))
        torch.save({'state_dict': dis.module.state_dict()}, os.path.join(self.folder_manager.config.cache_dir, 'D.pth'))
        # exit()
                    
    
    def _sample(self, batch_size):
        args = self.args
        z = torch.randn((batch_size, args.z_dim), device=args.device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=args.device)
        fake = self.G(z, y)
        return z, y, fake
    
    def _max_margin_loss(self, out, iden):
        real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
        tmp1 = torch.argsort(out, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
        margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

        return (-1 * real).mean() + margin.mean()
    
    def before_train(self):
        super().before_train()
        _, labels, fake = self._sample(5)
        print(fake.shape)
        labels = labels.cpu().tolist()
        fake = fake.cpu()
        # torch.save(f'epoch_{self.epoch}_{labels[0]}_{labels[1]}_{labels[2]}_{labels[3]}_{labels[4]}.png')
        import torchvision
        save_dir = os.path.join(self.folder_manager.config.cache_dir, 'train_sample')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch_{self.epoch}_{labels[0]}_{labels[1]}_{labels[2]}_{labels[3]}_{labels[4]}.png')
        torchvision.utils.save_image(fake, save_path, nrow=5, normalize=True)
    
    def train_gen_step(self, batch):
        
        
        args = self.args
        bs = len(batch[0])
        _, labels, fake = self._sample(bs)
        dis_loss = - self.D(fake).mean()
        
        aug_fake = args.augment(fake) if args.augment else fake
        
        pred = self.T(aug_fake).result
        inv_loss = self._max_margin_loss(pred, labels)
        
        loss = dis_loss + inv_loss * args.coef_inv_loss
        
        # print('aaa', bs)
        # while 1:
        #     pass
        
        super().loss_update(loss, self.optim_G)
        
        # return {
        #     'dis loss': dis_loss.item(),
        #     'inv loss': inv_loss.item(),
        #     'total loss': loss.item()
        # }
        return OrderedDict(
            dis_loss = dis_loss.item(),
            inv_loss = inv_loss.item(),
            total_loss = loss.item()
        )
        
    def train_dis_step(self, batch):
        args = self.args
        bs = len(batch[0])
        
        _, labels, fake = self._sample(bs)
        # print(fake.shape, labels.shape)
        dis_fake = self.D(fake, labels)
        dis_fake = torch.mean(torch.relu(1. + dis_fake))
        
        real_imgs, real_labels = batch
        real_imgs, real_labels = real_imgs.to(args.device), real_labels.to(args.device)
        # print(fake.shape, labels.shape)
        # print(real_imgs.shape, real_labels.shape)
        # print(real_labels[:5])
        dis_real = self.D(real_imgs, real_labels)
        dis_real = torch.mean(torch.relu(1. - dis_real))
        
        # exit()
        
        loss = dis_fake + dis_real
           
           
        super().loss_update(loss, self.optim_D)
        
        # return {
        #     'fake loss': dis_fake.item(),
        #     'real loss': dis_real.item(),
        #     'total loss': loss.item()
        # }
        return OrderedDict(
            fake_loss = dis_fake.item(),
            real_loss = dis_real.item(),
            total_loss = loss.item()
        )