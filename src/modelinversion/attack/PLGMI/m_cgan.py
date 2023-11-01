import argparse
import datetime
import json
import kornia
import os
import shutil
import torch
import torch.nn.functional as F
import torchvision

# from . import evaluation
from . import losses as L
from . import utils
from .dataset import FaceDataset, InfiniteSamplerWrapper, sample_from_data, sample_from_gen
from .models import inception
from ...models import get_model
from .models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from .models.generators.resnet64 import ResNetGenerator
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import shutil
from dataclasses import dataclass
from tqdm import tqdm
from ...utils import FolderManager, set_random_seed

@dataclass
class PlgmiCGanArgs:
    
    # dataset_dir: str
    dataset_name: str
    target_name: str
    # cache_dir: str
    batch_size: int
    loss_type: str
    inv_loss_type: str
    # gen_ckpt_dir: str
    
    device: str
    relativistic_loss: bool
    num_classes = 1000
    gen_num_features = 64
    gen_dim_z = 128
    gen_bottom_width = 4
    gen_distribution = 'normal'
    dis_num_features = 64
    lr = 0.0002
    beta1 = 0.
    beta2 = 0.9
    max_iteration = 30000
    n_dis = 5
    checkpoint_interval = 1000
    eval_interval = 1000
    alpha = 0.2 # weight of inv loss
    
class SimpleSampler:
    
    def __init__(self, loader) -> None:
        self.loader = loader
        
    def _sample(self):
        while True:
            for data in self.loader:
                yield data
                
    def sample(self, device):
        data = self._sample()
        return (d.to(device) for d in data)
    
def plgmi_train_cgan(
    target_name,
    dataset_name,
    dataset_dir,
    cache_dir,
    ckpt_dir,
    batch_size=64,
    loss_type = 'hinge', # hinge / dcgan
    inv_loss_type = 'margin', # ce / margin / poincare
    relative_loss = False
    # ,
    # device = 'cuda'
):
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, None)
    device = 'cuda'
    args = PlgmiCGanArgs(
        # dataset_dir, 
        dataset_name, 
        target_name, 
        # cache_dir, 
        batch_size,
        loss_type,
        inv_loss_type, 
        # gen_ckpt_dir=ckpt_dir,
        device=device,
        relativistic_loss=relative_loss
    )
    
    # if target_name.startswith("vgg16"):
    #     T = VGG16(1000)
    #     path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'VGG16_88.26.tar')
    # elif target_name.startswith('ir152'):
    #     T = IR152(1000)
    #     path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'IR152_91.16.tar')
    # elif target_name == "facenet64":
    #     T = FaceNet64(1000)
    #     path_T = os.path.join(ckpt_dir, 'target_eval', 'celeba', 'FaceNet64_88.50.tar')
    # T = (T).to(device)
    T = get_model(target_name, dataset_name, device=device)
    folder_manager.load_target_model_state_dict(T, dataset_name, target_name, device)
    T.eval()
    
    
    run(args, T, folder_manager)
    
def run(args: PlgmiCGanArgs, target_model, folder_manager: FolderManager):
    
    seed = 64
    set_random_seed(seed)
    
    device = args.device
    
    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img
    
    if args.dataset_name == 'celeba':
        # re_size = 64
        # crop_size = 108
        # offset_height = (218 - crop_size) // 2
        # offset_width = (178 - crop_size) // 2
        # crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        my_transform = _noise_adder
    elif args.dataset_name == 'ffhq':
        crop_size = 88
        offset_height = (128 - crop_size) // 2
        offset_width = (128 - crop_size) // 2
        re_size = 64
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        
        my_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(crop),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((re_size, re_size)),
            torchvision.transforms.ToTensor(),
            _noise_adder
        ])
        
    elif args.dataset_name == 'facescrub':
        re_size = 64
        crop_size = 64
        offset_height = (64 - crop_size) // 2
        offset_width = (64 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        
        my_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(crop),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((re_size, re_size)),
            torchvision.transforms.ToTensor(),
            _noise_adder
        ])
    else:
        print("Wrong Dataname!")
        
    
    
    top_n_selection_dir = os.path.join(folder_manager.config.cache_dir, 'top_n_selection', args.dataset_name, args.target_name)
    
    if not os.path.exists(top_n_selection_dir):
        print(f'dst dir: {top_n_selection_dir}')
        raise RuntimeError("please run top n selection first! ")
    
    train_set = ImageFolder(top_n_selection_dir, transform=my_transform)
    
    # train_loader = iter(DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu')))
    train_loader = iter(torch.utils.data.DataLoader(
        train_set, args.batch_size,
        sampler=InfiniteSamplerWrapper(train_set),
    ))
    
    # sampler = SimpleSampler(train_loader)
    
    # result_dir = os.path.join(args.cache_dir, 'train_cgan', args.dataset_name, )
    result_ckpt_dir = os.path.join(folder_manager.config.ckpt_dir, 'PLGMI')
    os.makedirs(result_ckpt_dir, exist_ok=True)
    
    result_D_path = os.path.join(result_ckpt_dir, f'{args.dataset_name}_{args.target_name.upper()}_PLG_MI_D.tar')
    result_G_path = os.path.join(result_ckpt_dir, f'{args.dataset_name}_{args.target_name.upper()}_PLG_MI_G.tar')
    
    train_img_gen_dir = os.path.join(folder_manager.config.cache_dir, 'train_cgan', 'train_img')
    os.makedirs(train_img_gen_dir, exist_ok=True)
    
    _n_cls = args.num_classes
    gen = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        activation=F.relu, num_classes=_n_cls, distribution=args.gen_distribution
    ).to(device)
    
    dis = SNResNetProjectionDiscriminator(args.dis_num_features, _n_cls, F.relu).to(device)
    
    opt_gen = torch.optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = torch.optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))
    
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)
    
    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    ).to(device)
    
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
                # fake image, fake label, sample z
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
                dis_fake = dis(fake, pseudo_y)
                # random transformation on the generated images
                # print(f'>> {fake.device}')
                # a = aug_list.parameters().
                # b = next(a)
                # c = b.device
                # print(f'>> {c}')
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
                    # print(">> aaaaaaaaa")
                    real, y = sample_from_data(args, device, train_loader)
                    # real, y = sampler.sample(device)
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

            # generate fake images
            fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
            # sample the real images
            # print(">> bbbaaaaaaaaa")
            real, y = sample_from_data(args, device, train_loader)
            # real, y = sampler.sample(device)
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

        # ==================== End of 1 iteration. ====================

        # if n_iter % args.log_interval == 0:
        #     print(
        #         'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
        #             n_iter, args.max_iteration, _l_g, cumulative_loss_dis, cumulative_inv_loss,
        #             cumulative_target_acc, ))
        #     # Save previews
        #     utils.save_images(
        #         n_iter, n_iter // args.checkpoint_interval, args.results_root,
        #         args.train_image_root, fake, real
        #     )
        if n_iter % args.checkpoint_interval == 0:
            # Save checkpoints!
            # utils.save_checkpoints(
            #     args, n_iter, n_iter // args.checkpoint_interval,
            #     gen, opt_gen, dis, opt_dis
            # )
            torch.save({'state_dict': gen.state_dict()}, result_G_path)
            torch.save({'state_dict': dis.state_dict()}, result_D_path)
            
    torch.save({'state_dict': gen.state_dict()}, result_G_path)
    torch.save({'state_dict': dis.state_dict()}, result_D_path)