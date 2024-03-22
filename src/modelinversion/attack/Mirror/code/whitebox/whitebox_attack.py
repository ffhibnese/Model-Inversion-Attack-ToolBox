
from dataclasses import dataclass
import os
import glob
import random

import torch
from torch import nn
from PIL import Image
import numpy as np

from ..utils import create_folder
from ..genforce.get_genforce import get_genforce
from ..utils.img_utils import clip, crop_img, resize_img, normalize, denormalize, clip_quantile_bound, get_input_resolution
from ..utils.acc import verify_acc
from ...code.classifiers.build_classifier import get_model
from ...code.select_w import find_closest_latent
from ...config import MirrorWhiteboxAttackConfig
from .....foldermanager import FolderManager
# @dataclass
# class MirrorWhiteBoxArgs:
#     arch_name: str
#     test_arch_name: str
#     genforce_model_name: str
#     # target_labels: list
#     gen_num_per_target: int
#     device: str
#     # calc_knn: bool
#     # batch_size: int
#     do_flip : bool = False
#     # use_cache : bool
#     loss_class_ce : float = 1
#     epoch_num : int = 5000
#     lr: float = 0.2
#     save_every : int = 100
#     use_dropout : bool = False
#     latent_space : str = 'w'
#     p_std_ce : int = 1
#     z_std_ce : int = 1
    
import math
    
def adjust_lr(optimizer: torch.optim.Optimizer, initial_lr, epoch, epochs, rampdown=0.25, rampup=0.05):
    # from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py#L45
    t = epoch / epochs
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    lr = initial_lr * lr_ramp

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
    
def mirror_white_box_attack(
    args: MirrorWhiteboxAttackConfig, 
    generator,
    target_net,
    eval_net,
    folder_manager: FolderManager,
    target_labels,
    to_target_transforms=None
):
    
    # if args.arch_name == 'sphere20a':
    #     # TODO: add sphere
    #     raise NotImplementedError('sphere20 is not implement')
    #     criterion = net_sphere.MyAngleLoss()
    # else:
    criterion = nn.CrossEntropyLoss()
    
    if args.genforce_name.startswith('stylegan'):
        use_z_plus_space = False
        use_w_space = 'w' in args.latent_space
        repeat_w = '+' not in args.latent_space
        if args.latent_space == 'z+':
            use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
            use_w_space = True
        if args.genforce_name.endswith('1024'):
            w_num_layers = 18
        elif args.genforce_name.endswith('512'):
            w_num_layers = 16
        else:
            w_num_layers = 14
            
        use_pre_generated_latents = True # args.use_cache
        # if use_pre_generated_latents:
        #     raise NotImplementedError('cache is not build')
        use_w_mean = not use_pre_generated_latents
        
        invert_lrelu = nn.LeakyReLU(5.)
        lrelu = nn.LeakyReLU(0.2)
        
        use_p_space_bound = args.p_std_ce > 0. and use_w_space and not use_z_plus_space
        
        if use_p_space_bound:
        
            w_dir = os.path.join(folder_manager.config.presample_dir, 'w')
            
            all_ws_gen_files = sorted(glob.glob(os.path.join(w_dir, 'sample_*.pt')))
            
            all_w_mins_ls = []
            all_w_maxs_ls = []
            
            for ws_file in all_ws_gen_files:
            
                all_ws = torch.load(ws_file, map_location=args.device).detach().to(args.device)
                # print(f'all_ws.shape: {all_ws.shape}')
                all_ps = invert_lrelu(all_ws)
                all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
                all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
                all_p_mins = all_p_means - args.p_std_ce * all_p_stds
                all_p_maxs = all_p_means + args.p_std_ce * all_p_stds
                all_w_mins = lrelu(all_p_mins)
                all_w_maxs = lrelu(all_p_maxs)
                all_w_mins_ls.append(all_w_mins)
                all_w_maxs_ls.append(all_w_maxs)
            all_w_mins = torch.mean(torch.cat(all_w_mins_ls, dim=0), dim=0)
            all_w_maxs = torch.mean(torch.cat(all_w_maxs_ls, dim=0), dim=0)
            
        #     print(all_w_mins.shape)
        #     print((all_w_maxs - all_w_mins).sum(dim=-1).mean())
        #     exit()
            
        # print('maomao')
        # exit()
        
    else:
        raise NotImplementedError(f'model {args.genforce_name} is not implented')
    
    # generator, _ = get_genforce(args.genforce_name, device=args.device, checkpoint_dir=folder_manager.config.ckpt_dir, use_w_space=use_w_space, use_z_plus_space=use_z_plus_space, repeat_w=repeat_w, use_discri=False)
    
    target_list = target_labels
    if isinstance(target_labels, torch.Tensor):
        target_list = target_labels.cpu().numpy().reshape(-1).tolist()
    
    # assert args.batch_size % len(target_list) == 0, f'batchsize: {args.batch_size} len target list: {len(target_list)}'
    nrow = args.gen_num_per_target
    dt = []
    for t in target_list:
        for _ in range(nrow):
            dt.append(t)
    targets = torch.LongTensor(dt).to(args.device)
    
    batch_size = len(target_list)
    # targets = torch.LongTensor(target_list).to(args.device)
    
    latent_dim = 512
    
    if use_pre_generated_latents:
        assert args.latent_space == 'w'
        # raise NotImplementedError('cache is not implemented')
        
        w_dict = find_closest_latent(target_net, args.device, target_list, nrow, args.target_name, folder_manager.config.presample_dir, to_target_transforms=to_target_transforms)[0]
        inputs = torch.cat([w_dict[t] for t in target_list], dim=0).to(args.device)
    
    else:
        inputs = torch.randn(batch_size, latent_dim)
        
    
        # TODO: check if has `else`
        if use_w_space:
            if use_z_plus_space:
                inputs = torch.randn(batch_size*w_num_layers, latent_dim)
                
            else:
                if use_w_mean:
                    # NOTE: we can try using model's w_avg buffer as the w_mean instead of sampling
                    n_mean_latent = 10000

                    with torch.no_grad():
                        latent_z_inputs = torch.randn(n_mean_latent, latent_dim, device=args.device)
                        latent_w = generator.G.mapping(latent_z_inputs)['w']
                        print(f'latent_z_inputs.shape: {latent_z_inputs.shape}')
                        print(f'latent_w.shape: {latent_w.shape}')
                        latent_w_mean = latent_w.mean(0)
                        # latent_w_std = ((latent_w - latent_w_mean).pow(2).sum() / n_mean_latent) ** 0.5
                        print(f'latent_w_mean.shape: {latent_w_mean.shape}')

                # if use_w_mean:
                    # optimize the w space instead of z space
                    inputs = latent_w_mean.detach().clone().repeat(batch_size, 1)
                else:
                    with torch.no_grad():
                        latent_z_inputs = torch.randn(batch_size, latent_dim, device=args.device)
                        latent_w = generator.G.mapping(latent_z_inputs)['w']
                        print(f'latent_z_inputs.shape: {latent_z_inputs.shape}')
                        print(f'latent_w.shape: {latent_w.shape}')
                    # optimize the w space instead of z space
                    inputs = latent_w.detach().clone()
                    
                if not repeat_w:
                    inputs = inputs.unsqueeze(1).repeat(1, w_num_layers, 1)  # shape: batch_size x num_layers x 512
    
    # with torch.no_grad():
    #     init_images = generator(inputs.to(args.device))
    #     # TODO: save image
    #     save_image(init_images,
    #                f'{args.tmp_image_dir}/images/output_{0:05d}.png',
    #                nrow=nrow)
    #     torch.save(init_images,
    #                f'{args.tmp_image_dir}/images/output_{0:05d}.pt')
    #     torch.save(inputs,
    #                f'{args.tmp_image_dir}/images/latent_input_{0:05d}.pt')

    origin_inputs = inputs.detach().clone()
    
    inputs.requires_grad_(True)
    
    optimizer = torch.optim.Adam([inputs], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    
    best_cost = 1e5
    for epoch in range(1, args.epoch_num + 1):
        _lr = adjust_lr(optimizer, args.lr, epoch, args.epoch_num)
        
        fake = generator(inputs)
        fake = crop_img(fake, args.target_name)
        
        input_images = normalize(fake*255., args.target_name)
        
        # horizontal flipping
        flip = random.random() > 0.5
        if args.do_flip and flip:
            input_images = torch.flip(input_images, dims=(3,))
        
        optimizer.zero_grad()
        # generator.zero_grad()
        
        # TODO: output
        if to_target_transforms is not None:
            input_images = to_target_transforms(input_images)
        outputs = target_net(input_images.to(args.device)).result
        
        loss_class = criterion(outputs, targets.to(args.device))
        
        loss = args.loss_class_ce * loss_class
        
        
        
        loss.backward()
        optimizer.step()
        
        if use_p_space_bound:
            inputs.data = clip_quantile_bound(inputs.data, all_w_mins, all_w_maxs)
            
        if best_cost > loss.item() or epoch == 0:
            best_inputs = inputs.data.clone()
            best_epoch = epoch
            best_cost = loss.item()
            
        if epoch % args.save_every == 0:
            print(f'---------epoch: {epoch}----------')
            print(f'lr: {_lr}\t class loss: {loss_class}')
            
            with torch.no_grad():
                fake = generator(inputs.detach().to(args.device))
                fake = crop_img(fake, args.target_name)
                target_fake = normalize(fake*255., args.target_name)
                if to_target_transforms is not None:
                    target_fake = to_target_transforms(target_fake)
                target_acc =  verify_acc(target_fake, targets, target_net, args.target_name)
                print(f'target_acc: {target_acc:.6f}')
                
                eval_fake = normalize(fake*255., args.target_name)
                
                if to_target_transforms is not None:
                    eval_fake = to_target_transforms(eval_fake)
                    
                eval_acc =  verify_acc(eval_fake, targets, eval_net, args.eval_name)
                print(f'eval_acc: {eval_acc:.6f}')
                
                # torch.save(inputs,
                #        f'{args.tmp_image_dir}/images/latent_input_{epoch//args.save_every:05d}.pt')
                # save_image(denormalize(target_fake, args.arch_name),
                #         f'{args.tmp_image_dir}/images/output_{epoch//args.save_every:05d}.png',
                #         nrow=nrow)

    with torch.no_grad():   
        latent_inputs = best_inputs.detach().clone()
        fake = generator(latent_inputs.detach().to(args.device))
        # don't resize and downsample the images, but save the high-resolution images
        fake = normalize(fake*255., args.target_name)
        
        final_acc = verify_acc(eval_fake, targets, eval_net, args.target_name)
        # print(f'final acc: {eval_acc:.6f}')
        
        if to_target_transforms is not None:
            fake = to_target_transforms(fake)

    for i in range(fake.shape[0]):
        target = targets[i].item()

        image = denormalize(fake[i], args.target_name) #.data.cpu().numpy().transpose((1, 2, 0))
        # pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        # pil_image.save(save_filename)
        # if to_target_transforms is not None:
        #     image = to_target_transforms(image)
            
        folder_manager.save_result_image(image, target)

    # torch.save(latent_inputs, f'{args.final_image_dir}/latent_inputs.pt')
    return {'acc': final_acc}