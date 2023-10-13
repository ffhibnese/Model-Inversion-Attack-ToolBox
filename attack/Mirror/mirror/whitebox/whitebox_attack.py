import torch
from torch import nn
from dataclasses import dataclass
from attack.Mirror.utils import create_folder
import os
from attack.Mirror.genforce.get_genforce import get_genforce
from torchvision.utils import save_image
from attack.Mirror.utils.img_utils import clip, crop_img, resize_img, normalize, denormalize, clip_quantile_bound, get_input_resolution
from attack.Mirror.utils.acc import verify_acc
from PIL import Image
import numpy as np
import random
from attack.Mirror.mirror.classifiers.build_classifier import get_model
from attack.Mirror.mirror.select_w import find_closest_latent
import glob
from metrics.knn import get_knn_dist

@dataclass
class MirrorWhiteBoxArgs:
    arch_name: str
    test_arch_name: str
    genforce_model_name: str
    target_labels: list
    work_dir: str
    checkpoint_dir: str
    do_flip : bool
    use_cache : bool
    loss_class_ce : float
    epoch_num : int
    batch_size : int
    lr: float
    save_every : int
    use_dropout : bool
    latent_space : str
    p_std_ce : int
    z_std_ce : int
    device: str
    pre_sample_dir: str
    
    final_image_dir: str
    tmp_image_dir: str
    image_resolution: int
    test_image_resolution: int
    
    calc_knn: bool

    

def mirror_white_box_attack(
    arch_name,
    test_arch_name,
    genforce_model_name,
    target_labels,
    work_dir,
    checkpoint_dir,
    classifier_dir,
    dataset_name,
    pre_sample_dir = None,
    use_cache = False,
    do_flip = False,
    loss_class_ce = 1,
    epoch_num = 5000,
    batch_size = 8,
    lr = 0.2,
    save_every = 100,
    use_dropout = False,
    latent_space = 'w',
    p_std_ce = 1,
    z_std_ce = 1,
    device = 'cuda',
    calc_knn = False
):
    if genforce_model_name != 'stylegan2_ffhq1024':
        torch.backends.cudnn.benchmark = True
        
    final_image_dir = os.path.join(work_dir, 'final_images')
    tmp_image_dir = os.path.join(work_dir, 'generations')
    
    create_folder(final_image_dir)
    create_folder(tmp_image_dir)
    create_folder(os.path.join(tmp_image_dir, 'images'))
    
    target_net: nn.Module = get_model(arch_name, device=device, use_dropout=use_dropout, classifier_dir=classifier_dir, dataset_name=dataset_name)
    eval_net: nn.Module = get_model(test_arch_name, device=device, use_dropout=use_dropout, classifier_dir=classifier_dir, dataset_name=dataset_name)
    image_resolution = get_input_resolution(arch_name)
    test_image_resolution = get_input_resolution(test_arch_name)
    
        
    args = MirrorWhiteBoxArgs(
        arch_name=arch_name,
        test_arch_name=test_arch_name,
        genforce_model_name=genforce_model_name,
        target_labels=target_labels,
        work_dir=work_dir,
        checkpoint_dir=checkpoint_dir,
        do_flip=do_flip,
        use_cache=use_cache,
        loss_class_ce=loss_class_ce,
        epoch_num=epoch_num, 
        batch_size=batch_size,
        save_every=save_every,
        use_dropout=use_dropout,
        latent_space=latent_space,
        p_std_ce=p_std_ce,
        z_std_ce=z_std_ce,
        device=device,
        final_image_dir=final_image_dir,
        tmp_image_dir=tmp_image_dir,
        image_resolution=image_resolution,
        test_image_resolution=test_image_resolution,
        lr=lr,
        pre_sample_dir=pre_sample_dir,
        calc_knn=calc_knn
    )
    
    # run
    run(args, target_net, eval_net)
    
import math
    
def adjust_lr(optimizer, initial_lr, epoch, epochs, rampdown=0.25, rampup=0.05):
    # from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py#L45
    t = epoch / epochs
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    lr = initial_lr * lr_ramp

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
    
def run(
    args: MirrorWhiteBoxArgs, 
    target_net,
    eval_net
):
    
    if args.arch_name == 'sphere20a':
        # TODO: add sphere
        raise NotImplementedError('sphere20 is not implement')
        criterion = net_sphere.MyAngleLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if args.genforce_model_name.startswith('stylegan'):
        use_z_plus_space = False
        use_w_space = 'w' in args.latent_space
        repeat_w = '+' not in args.latent_space
        if args.latent_space == 'z+':
            use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
            use_w_space = True
        if args.genforce_model_name.endswith('1024'):
            w_num_layers = 18
        elif args.genforce_model_name.endswith('512'):
            w_num_layers = 16
        else:
            w_num_layers = 14
            
        use_pre_generated_latents = args.use_cache
        # if use_pre_generated_latents:
        #     raise NotImplementedError('cache is not build')
        use_w_mean = not use_pre_generated_latents
        
        invert_lrelu = nn.LeakyReLU(5.)
        lrelu = nn.LeakyReLU(0.2)
        
        use_p_space_bound = args.p_std_ce > 0. and use_w_space and not use_z_plus_space
        
        if use_p_space_bound:
        
            w_dir = os.path.join(args.pre_sample_dir, 'w')
            
            all_ws_gen_files = sorted(glob.glob(os.path.join(w_dir, 'sample_*.pt')))
            
            all_w_mins_ls = []
            all_w_maxs_ls = []
            
            for ws_file in all_ws_gen_files:
            
                all_ws = torch.load(ws_file).detach().to(args.device)
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
            all_w_mins = torch.mean(torch.cat(all_w_mins_ls, dim=0))
            all_w_maxs = torch.mean(torch.cat(all_w_maxs_ls, dim=0))
        
        
    else:
        raise NotImplementedError(f'model {args.genforce_model_name} is not implented')
    
    generator, _ = get_genforce(args.genforce_model_name, device=args.device, checkpoint_dir=args.checkpoint_dir, use_w_space=use_w_space, use_z_plus_space=use_z_plus_space, repeat_w=repeat_w, use_discri=False)
    
    target_list = args.target_labels
    
    assert args.batch_size % len(target_list) == 0
    nrow = args.batch_size // len(target_list)
    dt = []
    for t in target_list:
        for _ in range(nrow):
            dt.append(t)
    targets = torch.LongTensor(dt).to(args.device)
    
    latent_dim = 512
    
    if use_pre_generated_latents:
        assert args.latent_space == 'w'
        # raise NotImplementedError('cache is not implemented')
        
        w_dict = find_closest_latent(target_net, args.image_resolution, target_list, nrow, args.arch_name, args.pre_sample_dir)[0]
        inputs = torch.cat([w_dict[t] for t in target_list], dim=0).to(args.device)
    
    else:
        inputs = torch.randn(args.batch_size, latent_dim)
        
    
        # TODO: check if has `else`
        if use_w_space:
            if use_z_plus_space:
                inputs = torch.randn(args.batch_size*w_num_layers, latent_dim)
                
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
                    inputs = latent_w_mean.detach().clone().repeat(args.batch_size, 1)
                else:
                    with torch.no_grad():
                        latent_z_inputs = torch.randn(args.batch_size, latent_dim, device=args.device)
                        latent_w = generator.G.mapping(latent_z_inputs)['w']
                        print(f'latent_z_inputs.shape: {latent_z_inputs.shape}')
                        print(f'latent_w.shape: {latent_w.shape}')
                    # optimize the w space instead of z space
                    inputs = latent_w.detach().clone()
                    
                if not repeat_w:
                    inputs = inputs.unsqueeze(1).repeat(1, w_num_layers, 1)  # shape: batch_size x num_layers x 512
    
    with torch.no_grad():
        init_images = generator(inputs.to(args.device))
        # TODO: save image
        save_image(init_images,
                   f'{args.tmp_image_dir}/images/output_{0:05d}.png',
                   nrow=nrow)
        torch.save(init_images,
                   f'{args.tmp_image_dir}/images/output_{0:05d}.pt')
        torch.save(inputs,
                   f'{args.tmp_image_dir}/images/latent_input_{0:05d}.pt')
        
        
    inputs = inputs.detach().clone().to(args.device)
    print(f'----------input is leaf: {inputs.is_leaf}')


    origin_inputs = inputs.detach().clone()
    
    inputs.requires_grad_(True)
    
    print(f'----------input is leaf: {inputs.is_leaf}')
    
    optimizer = torch.optim.Adam([inputs], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    
    best_cost = 1e5
    for epoch in range(1, args.epoch_num + 1):
        _lr = adjust_lr(optimizer, args.lr, epoch, args.epoch_num)
        
        fake = generator(inputs)
        fake = crop_img(fake, args.arch_name)
        
        input_images = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
        
        # horizontal flipping
        flip = random.random() > 0.5
        if args.do_flip and flip:
            input_images = torch.flip(input_images, dims=(3,))
        
        optimizer.zero_grad()
        generator.zero_grad()
        
        # TODO: output
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
                fake = crop_img(fake, args.arch_name)
                target_fake = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
                target_acc =  verify_acc(target_fake, targets, target_net, args.arch_name)
                print(f'target_acc: {target_acc:.6f}')
                
                eval_fake = normalize(resize_img(fake*255., args.test_image_resolution), args.arch_name)
                eval_acc =  verify_acc(eval_fake, targets, eval_net, args.test_arch_name)
                print(f'eval_acc: {eval_acc:.6f}')
                
                torch.save(inputs,
                       f'{args.tmp_image_dir}/images/latent_input_{epoch//args.save_every:05d}.pt')
                save_image(denormalize(target_fake, args.arch_name),
                        f'{args.tmp_image_dir}/images/output_{epoch//args.save_every:05d}.png',
                        nrow=nrow)

    with torch.no_grad():   
        latent_inputs = best_inputs.detach().clone()
        fake = generator(latent_inputs.detach().to(args.device))
        # don't resize and downsample the images, but save the high-resolution images
        fake = normalize(fake*255., args.arch_name)

    for i in range(fake.shape[0]):
        target = targets[i].item()
        # save_filename = f'{args.final_image_dir}/img_label{target:05d}_id{i:03d}_iter{best_epoch}.jpg'
        save_dirname = os.path.join(args.final_image_dir, f'{target}')
        os.makedirs(save_dirname, exist_ok=True)
        save_filename = os.path.join(save_dirname, f'{i}.jpg')

        torch.save(latent_inputs[i], save_filename[:-4]+'.pt')

        image_np = denormalize(fake[i], args.arch_name).data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(save_filename)

    torch.save(latent_inputs, f'{args.final_image_dir}/latent_inputs.pt')
    
    if args.calc_knn:
        feat_dir = os.path.join(args.checkpoint_dir, "PLG_MI", "celeba_private_feats")
        knn_dist = get_knn_dist(eval_net, args.final_image_dir, feat_dir, resolution=112)
        print(f"knn dist: {knn_dist}")