import collections
import glob
import math
import os
import random

from scipy.stats import truncnorm
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image
import torchvision.transforms.functional as F

from genforce import my_get_GD

from my_utils import clip, crop_img, resize_img, normalize, denormalize, clip_quantile_bound
from my_concat_final_images import concat_final_images


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


def verify_accuracy(input, target, model, arch_name):
    def _split_into_even(data, n):
        assert len(data) % n == 0
        _s = len(data)//n
        _r = []
        for i in range(n):
            _ss = _s * i
            _r.append(data[_ss:_ss+_s])
        return _r

    def accuracy(output, target):
        batch_size = target.size(0)
        _, pred = output.max(dim=1)
        acc = ((pred == target).sum()*100./batch_size)
        return acc

    with torch.no_grad():
        device = next(model.parameters()).device
        n = 1
        inputs = _split_into_even(input, n)
        targets = _split_into_even(target, n)
        confidence_str = []
        acc = 0
        for input, target in zip(inputs, targets):
            if arch_name == 'sphere20a':
                output = model(input.to(device))[0]
            else:
                output = model(input.to(device))
            confidence = nn.functional.softmax(output, dim=1)
            assert n == 1, 'the following loop requires n == 1, or we need to recompute the i'
            for i, t in enumerate(target):
                confidence_str.append(f'{confidence[i][t]:.6f}')
            acc += accuracy(output.data, target.to(device)).item()

    print('Verifier accuracy:', acc/n)
    print('Confidence:', ','.join(confidence_str))


@torch.no_grad()
def find_closest_latents(target_model, image_resolution, use_genforce, genforce_model, targets_list, k, arch_name, all_ws, args):
    device = next(target_model.parameters()).device
    target_ranked_confidence_dict = collections.defaultdict(list)

    if args.use_cache:
        is_zplus_space = 'zplus' in args.all_ws_pt_file
        is_z_space = not is_zplus_space and 'z' in args.all_ws_pt_file
        all_logits_file = os.path.join('./blackbox_attack_data/stylegan',
                                       arch_name,
                                       'use_dropout' if args.use_dropout else 'no_dropout',
                                       'all_logits.pt')
        all_logits = torch.load(all_logits_file).to(args.device)
        all_prediction = nn.functional.softmax(all_logits, dim=1)
        target_ranked_confidence_dict = {t: all_prediction[:, t] for t in targets_list}
    else:
        all_img_gen_files = sorted(glob.glob(os.path.join(args.pre_samples_dir, 'sample_*_img.pt')))
        assert (len(all_img_gen_files)) > 0
        is_zplus_space = 'zplus' in all_img_gen_files[0]
        is_z_space = not is_zplus_space and 'z' in all_img_gen_files[0]
        # find the highest confidence
        for img_gen_file in all_img_gen_files:
            fake = torch.load(img_gen_file).to(device)
            fake = crop_img(fake, arch_name)
            fake = normalize(resize_img(fake*255., image_resolution), arch_name)
            outputs = []
            for i in range(0, len(fake), 50):
                if arch_name == 'sphere20a':
                    outputs.append(target_model(fake[i:i+50])[0])
                else:
                    outputs.append(target_model(fake[i:i+50]))
            outputs = torch.cat(outputs, dim=0)
            outputs = nn.functional.softmax(outputs, dim=1)
            for t in targets_list:
                t_out = outputs[:, t]
                target_ranked_confidence_dict[t].append(t_out)
        target_ranked_confidence_dict = {t: torch.cat(v, dim=0) for t, v in target_ranked_confidence_dict.items()}

    target_topk_ind_dict = {}
    for t, v in target_ranked_confidence_dict.items():
        topk_conf, topk_ind = torch.topk(v, k, dim=0, largest=True, sorted=True)
        print(f'{t}: {topk_conf}\t{topk_ind}')
        target_topk_ind_dict[t] = topk_ind.tolist()

    if args.use_cache and all_ws is not None:
        return_latent_indices = []
        for t in targets_list:
            for i in target_topk_ind_dict[t]:
                return_latent_indices.append(i)
        return_latents = all_ws[return_latent_indices].detach().clone()
    else:
        all_latent_files = [x[:-6]+'latent.pt' for x in all_img_gen_files]
        all_latents = []
        for latent_file in all_latent_files:
            all_latents.append(torch.load(latent_file).to(device))
        all_latents = torch.cat(all_latents, dim=0)
        print(f'all_latents.shape: {all_latents.shape}')
        return_latents = []
        return_latent_indices = []
        for t in targets_list:
            for i in target_topk_ind_dict[t]:
                return_latents.append(all_latents[i])
                return_latent_indices.append(i)
        return_latents = torch.stack(return_latents, dim=0)

    print(f'return_latents.shape: {return_latents.shape}')
    assert (is_zplus_space and not is_z_space) or (not is_zplus_space and is_z_space)
    return return_latents, is_z_space, is_zplus_space, return_latent_indices


def mirror_attack(args, target_model, verifier_model):

    p_space_bound_std_ce=args.p_std_ce

    best_cost = 1e4

    use_genforce = True
    genforce_model = args.genforce_model

    if genforce_model.startswith('stylegan'):
        use_z_plus_space = False
        if genforce_model in ['stylegan_celeba_partial256', 'stylegan_ffhq256', 'stylegan2_ffhq1024', 'stylegan_cat256', 'stylegan_animeportrait512', 'stylegan_animeface512', 'stylegan_artface512', 'stylegan_car512', ]:
            # use_w_space = True
            # repeat_w = True
            # use_w_mean = True
            use_w_space = 'w' in args.latent_space
            repeat_w = '+' not in args.latent_space
            if args.latent_space == 'z+':
                use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
                use_w_space = True
            use_loss_w_mean = False
            w_num_layers = 14  # 18 for img 1024x1024, 14 for img 256x256
            if genforce_model.endswith('1024'):
                w_num_layers = 18
            elif genforce_model.endswith('512'):
                w_num_layers = 16
            use_discri = args.loss_discri_ce > 0.
            use_loss_latent = False
            use_pre_generated_latents = args.use_cache or (args.pre_samples_dir and os.path.isdir(args.pre_samples_dir))
            use_w_mean = not use_pre_generated_latents and args.use_w_mean
            normalize_z_vector = False
            bound_latent_vector = False
            use_loss_l2_bound_latent = args.loss_l2_bound_latent_ce > 0.
            trunc_psi = args.trunc_psi
            trunc_layers = args.trunc_layers
            substitute_w_avg = False
            all_ws = None
            all_ws_pt_file = args.all_ws_pt_file
            use_p_space_bound = args.p_std_ce > 0. and use_w_space and not use_z_plus_space
            to_truncate = args.to_truncate_z and (use_z_plus_space or not use_w_space) and (not normalize_z_vector)
            if args.naive_clip_w_bound > 0.:
                use_p_space_bound = True
            if args.energy:
                use_p_space_bound = False
        else:
            raise AssertionError('wrong stylegan model')

        invert_lrelu = nn.LeakyReLU(negative_slope=5.)
        lrelu = nn.LeakyReLU(negative_slope=0.2)
        if use_p_space_bound:
            if args.naive_clip_w_bound > 1000:
                # assume w is normal distribution
                all_ws = torch.load(all_ws_pt_file).detach().to(args.device)
                print(f'all_ws.shape: {all_ws.shape}')
                all_ps = all_ws
                all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
                all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
                all_p_mins = all_p_means - p_space_bound_std_ce * all_p_stds
                all_p_maxs = all_p_means + p_space_bound_std_ce * all_p_stds
                all_w_mins = all_p_mins
                all_w_maxs = all_p_maxs
            elif args.naive_clip_w_bound > 100.:
                # only keep the central percent part
                all_ws = torch.load(all_ws_pt_file).detach().to(args.device)
                print(f'all_ws.shape: {all_ws.shape}')
                each_ws_list = [sorted(all_ws[:, i].tolist()) for i in range(all_ws.shape[1])]
                percent = args.naive_clip_w_bound/100 - 1
                all_w_ind = int(all_ws.shape[0] * (1+percent)/2)
                all_w_maxs = torch.tensor([x[all_w_ind] for x in each_ws_list]).to(args.device).unsqueeze(0)
                all_w_ind = int(all_ws.shape[0] * (1-percent)/2)
                all_w_mins = torch.tensor([x[all_w_ind] for x in each_ws_list]).to(args.device).unsqueeze(0)
            elif args.naive_clip_w_bound > 0.:
                all_w_maxs = torch.ones(1, 512).to(args.device) * args.naive_clip_w_bound
                all_w_mins = -1. * all_w_maxs.clone()
            else:
                all_ws = torch.load(all_ws_pt_file).detach().to(args.device)
                print(f'all_ws.shape: {all_ws.shape}')
                all_ps = invert_lrelu(all_ws)
                all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
                all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
                all_p_mins = all_p_means - p_space_bound_std_ce * all_p_stds
                all_p_maxs = all_p_means + p_space_bound_std_ce * all_p_stds
                all_w_mins = lrelu(all_p_mins)
                all_w_maxs = lrelu(all_p_maxs)
            print(f'all_w_mins.shape: {all_w_mins.shape}')

        if args.energy:
            all_ws = torch.load(all_ws_pt_file).detach()
            print(f'all_ws.shape: {all_ws.shape}')
            all_ps = invert_lrelu(all_ws)
            all_p_cov = (torch.from_numpy(np.linalg.inv(np.cov(all_ps.cpu().numpy().T))).float().to(args.device)).unsqueeze(0).expand(args.bs, -1, -1)
            all_p_means = torch.mean(all_ps, dim=0, keepdim=True).to(args.device)
            print(f'all_p_cov.shape: {all_p_cov.shape}, all_p_means.shape: {all_p_means.shape}')
    elif genforce_model == 'pggan_celebahq1024':
        use_z_plus_space = False
        use_w_space = False
        use_w_mean = False
        repeat_w = False
        use_loss_w_mean = False
        w_num_layers = 0
        use_discri = False
        use_loss_latent = False
        use_pre_generated_latents = False
        normalize_z_vector = False
        bound_latent_vector = False
        use_loss_l2_bound_latent = False
        trunc_psi = 0
        trunc_layers = 0
        substitute_w_avg = False
        all_ws = None
        all_ws_pt_file = None
        use_p_space_bound = False
        to_truncate = True

    generator, discri = my_get_GD.main(args.device, genforce_model, args.bs, args.bs, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space, trunc_psi=trunc_psi, trunc_layers=trunc_layers)

    assert isinstance(args.targets, list)
    targets_list = args.targets

    args.nrow = int(args.bs / len(args.targets))

    # Make the same target adjacent
    dt = []
    for t in args.targets:
        for _ in range(int(args.bs / len(args.targets))):
            dt.append(t)
    args.targets = dt

    targets = torch.LongTensor(args.targets * (int(args.bs / len(args.targets)))).to('cuda')

    # def truncated_z_sample(batch_size, z_dim, truncation=0.25, seed=None):
    def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
        return truncation * values

    print(f'use_genforce = {use_genforce}\n'
          f'use_w_space = {use_w_space}\n'
          f'use_z_plus_space = {use_z_plus_space}\n'
          f'use_w_mean = {use_w_mean}\n'
          f'repeat_w = {repeat_w}\n'
          f'w_num_layers = {w_num_layers}\n'
          f'use_discri = {use_discri}\n'
          f'use_loss_w_mean = {use_loss_w_mean}\n'
          f'use_loss_latent = {use_loss_latent}\n'
          f'use_pre_generated_latents = {use_pre_generated_latents}\n'
          f'to_truncate = {to_truncate}\n'
          f'normalize_z_vector = {normalize_z_vector}\n'
          f'bound_latent_vector = {bound_latent_vector}\n'
          f'use_loss_l2_bound_latent = {use_loss_l2_bound_latent}\n'
          f'substitute_w_avg = {substitute_w_avg}\n'
          f'trunc_psi = {trunc_psi}, trunc_layers = {trunc_layers}\n'
          f'use_p_space_bound = {use_p_space_bound} mean+-{p_space_bound_std_ce}*std\n'
          f'use_naive_clip_w_bound = {args.naive_clip_w_bound}\n'
          f'energy = {args.energy}')

    latent_dim = 512

    if use_pre_generated_latents:
        inputs, is_z_space, is_zplus_space, return_latent_indices = find_closest_latents(target_model, args.image_resolution, use_genforce, genforce_model, targets_list, args.nrow, args.arch_name, all_ws, args)
        if is_z_space:
            print('load z pregenerated images')
            if use_w_space:
                if use_z_plus_space:
                    inputs = inputs.unsqueeze(1).repeat(1, w_num_layers, 1)  # shape: bs x num_layers x 512
                    inputs = inputs.view(-1, latent_dim)
                else:
                    with torch.no_grad():
                        if all_ws is None:
                            if args.use_cache:
                                raise AssertionError('Should not reach here when using the cache.')
                            latent_w = generator.G.mapping(inputs)['w'].detach().clone()
                        else:
                            if args.use_cache:
                                latent_w = inputs
                            else:
                                latent_w = all_ws[return_latent_indices].detach().clone()

                        if trunc_psi == 1. or trunc_layers == 0:
                            # NOTE: current presampled images are using 0.7 and 8 as truncation parameters
                            inputs = generator.G.truncation(latent_w, 0.7, 8)
                        else:
                            inputs = latent_w

                        # substitue the w_avg
                        if substitute_w_avg:
                            # when generating the image pairs, trunc_psi=0.7, trunc_layers=8
                            latent_w = generator.G.truncation(latent_w, 0.7, 8)
                            inputs = latent_w
                            assert repeat_w, 'currently we only support repeat_w=True w space mode'
                            print('substitute w_avg')
                            generator.G.truncation.w_avg = inputs.detach().clone()

                    if not repeat_w:
                        inputs = inputs.unsqueeze(1).repeat(1, w_num_layers, 1)  # shape: bs x num_layers x 512
        elif is_zplus_space and use_w_space and not use_z_plus_space:
            print('load zplus pregenerated images')
            assert not repeat_w, 'we must use w+ space now'
            latent_w = generator.G.mapping(inputs.view(-1, latent_dim))['w'].detach().clone()
            inputs = latent_w.view_as(inputs)
            assert inputs.shape[0] == args.bs and inputs.shape[1] == w_num_layers and inputs.shape[2] == latent_dim

        # inputs = torch.load('./tmp/projected/gmi_stylegan_celeba_partial256_w+_from_real_0.1ce_10/batch_0_latent.pt')[:8]
        if use_loss_w_mean:
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

            latent_w_mean = latent_w_mean.detach().clone().unsqueeze(0)  # shape 1 x 512
            if not repeat_w:
                latent_w_mean = latent_w_mean.unsqueeze(0)  # shape 1 x 1 x 512
    else:
        # not pregen

        if to_truncate:
            inputs = torch.from_numpy(truncated_z_sample(args.bs, latent_dim, truncation=args.z_std_ce/2., seed=0)).float()
        else:
            # prepend gmi to deepinversion
            inputs = torch.randn(args.bs, latent_dim)

        if use_genforce and use_w_space:
            if use_z_plus_space:
                if to_truncate:
                    inputs = torch.from_numpy(truncated_z_sample(args.bs*w_num_layers, latent_dim, truncation=args.z_std_ce/2., seed=0)).float()
                else:
                    inputs = torch.randn(args.bs*w_num_layers, latent_dim)
            else:
                if use_w_mean or use_loss_w_mean:
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

                    if use_loss_w_mean:
                        latent_w_mean = latent_w_mean.detach().clone().unsqueeze(0)  # shape 1 x 512

                if use_w_mean:
                    # optimize the w space instead of z space
                    inputs = latent_w_mean.detach().clone().repeat(args.bs, 1)
                else:
                    with torch.no_grad():
                        latent_z_inputs = torch.randn(args.bs, latent_dim, device=args.device)
                        latent_w = generator.G.mapping(latent_z_inputs)['w']
                        print(f'latent_z_inputs.shape: {latent_z_inputs.shape}')
                        print(f'latent_w.shape: {latent_w.shape}')
                    # optimize the w space instead of z space
                    inputs = latent_w.detach().clone()

                if not repeat_w:
                    inputs = inputs.unsqueeze(1).repeat(1, w_num_layers, 1)  # shape: bs x num_layers x 512
                    if use_loss_w_mean:
                        latent_w_mean = latent_w_mean.unsqueeze(0)  # shape 1 x 1 x 512

    with torch.no_grad():
        init_images = generator(inputs.to(args.device))
        save_image(init_images,
                   f'{args.tmp_img_dirname}/images/output_{0:05d}.png',
                   nrow=args.nrow)
        torch.save(init_images,
                   f'{args.tmp_img_dirname}/images/output_{0:05d}.pt')
        torch.save(inputs,
                   f'{args.tmp_img_dirname}/images/latent_input_{0:05d}.pt')

    inputs = inputs.to(args.device)

    origin_inputs = inputs.detach().clone()

    inputs.requires_grad_(True)

    if args.energy:
        # args.epochs = 1000
        pass

    optimizer = optim.Adam([inputs, ], lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    for epoch in range(1, args.epochs+1):
        # learning rate scheduling
        _lr = adjust_lr(optimizer, args.lr, epoch, args.epochs)

        # perform downsampling if needed
        fake = generator(inputs.to(args.device))
        fake = crop_img(fake, args.arch_name)
        input_images = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)

        # horizontal flipping
        flip = random.random() > 0.5
        if args.do_flip and flip:
            input_images = torch.flip(input_images, dims=(3,))

        # forward pass
        optimizer.zero_grad()
        if use_discri:
            discri.zero_grad()
        generator.zero_grad()

        outputs = target_model(input_images.to(args.device))
        loss_class = args.criterion(outputs, targets.to(args.device))
        loss = args.loss_class_ce * loss_class

        loss_discri = 0.
        loss_discri_ce = args.loss_discri_ce
        if use_discri:
            fake_scores = discri(generator(inputs.to(args.device)))
            loss_discri = nn.functional.softplus(-fake_scores).mean() * loss_discri_ce
            loss = loss + loss_discri

        loss_latent_mean = 0.
        loss_latent_var = 0.
        loss_latent_mean_ce = 10000.
        loss_latent_var_ce = 10.
        if use_loss_latent:
            if use_z_plus_space or not use_w_space:
                inputs_mean = inputs.mean(dim=1)
                inputs_var = inputs.var(dim=1, unbiased=False)
                loss_latent_mean = loss_latent_mean_ce * torch.norm(inputs_mean, 2)
                loss_latent_var = loss_latent_var_ce * torch.norm(inputs_var-1., 2)
                loss = loss + loss_latent_mean + loss_latent_var

        loss_w_mean = 0.
        loss_w_mean_ce = 1. if use_loss_w_mean else 0.
        if use_w_space and use_loss_w_mean:
            loss_w_mean = torch.norm(inputs-latent_w_mean, 2)
            loss_w_mean = loss_w_mean_ce * loss_w_mean
            loss = loss + loss_w_mean

        loss_l2_bound_latent = 0.
        loss_l2_bound_latent_ce = args.loss_l2_bound_latent_ce
        if use_w_space and use_loss_l2_bound_latent:
            loss_l2_bound_latent = loss_l2_bound_latent_ce * torch.norm(inputs-origin_inputs, 2)
            loss = loss + loss_l2_bound_latent

        loss_energy = 0.
        loss_energy_ce = 1e-4
        if args.energy and use_w_space:
            v = (invert_lrelu(inputs) - all_p_means).unsqueeze(1)
            v1 = v.permute(0, 2, 1)
            loss_energy = loss_energy_ce * torch.bmm(torch.bmm(v, all_p_cov), v1).mean()
            loss = loss + loss_energy

        if epoch % args.save_every==0:
            print(f'------------ epoch {epoch}----------')
            print('lr', _lr)
            print('total loss', loss.item())
            print(f'class loss (multiplied by {args.loss_class_ce})', loss_class.item())
            print(f'loss_discri (multiplied by {loss_discri_ce})', loss_discri and loss_discri.item())
            print(f'loss_latent_mean (multiplied by {loss_latent_mean_ce})', loss_latent_mean and loss_latent_mean.item())
            print(f'loss_latent_var (multiplied by {loss_latent_var_ce})', loss_latent_var and loss_latent_var.item())
            print(f'loss_w_mean (multiplied by {loss_w_mean_ce})', loss_w_mean and loss_w_mean.item())
            print(f'loss_l2_bound_latent (multiplied by {loss_l2_bound_latent_ce})', loss_l2_bound_latent and loss_l2_bound_latent.item())
            print(f'loss_energy (multiplied by {loss_energy_ce})', loss_energy and loss_energy.item())

            if normalize_z_vector:
                _std = torch.std(inputs.data, unbiased=False).item()
                _mean = torch.mean(inputs.data).item()
                print(f'_std: {_std:.6f}, _mean: {_mean:.6f}')

            with torch.no_grad():
                fake = generator(inputs.detach().to(args.device))
                fake = crop_img(fake, args.arch_name)
                fake = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
                verify_accuracy(fake, targets, verifier_model, args.arch_name)

        loss.backward()

        optimizer.step()

        if to_truncate:
            inputs.data = torch.clamp(inputs.data, -args.z_std_ce, args.z_std_ce)
            # inputs.data = torch.clamp(inputs.data, -0.5, 0.5)
        elif normalize_z_vector:
            _std = torch.std(inputs.data, 1, unbiased=False, keepdim=True)
            _mean = torch.mean(inputs.data, 1, keepdim=True)
            inputs.data = (inputs.data-_mean)/_std
        elif bound_latent_vector:
            bound = 0.5
            diff = torch.clamp(inputs.data - origin_inputs, -bound, bound)
            inputs.data = origin_inputs + diff

        if use_p_space_bound:
            inputs.data = clip_quantile_bound(inputs.data, all_w_mins, all_w_maxs)

        if best_cost > loss.item() or epoch == 0:
            best_inputs = inputs.data.clone()
            best_epoch = epoch
            best_cost = loss.item()

        if epoch % args.save_every==0 and (args.save_every > 0):
            with torch.no_grad():
                fake = generator(inputs.detach().to(args.device))
                fake = crop_img(fake, args.arch_name)
                fake = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
            torch.save(inputs,
                       f'{args.tmp_img_dirname}/images/latent_input_{epoch//args.save_every:05d}.pt')
            save_image(denormalize(fake, args.arch_name),
                       f'{args.tmp_img_dirname}/images/output_{epoch//args.save_every:05d}.png',
                       nrow=args.nrow)

    with torch.no_grad():
        latent_inputs = best_inputs.detach().clone()
        fake = generator(best_inputs.detach().to(args.device))
        # don't resize and downsample the images, but save the high-resolution images
        fake = normalize(fake*255., args.arch_name)

    for i in range(fake.shape[0]):
        target = targets[i].item()
        save_filename = f'{args.final_img_dirname}/img_label{target:05d}_id{i:03d}_iter{best_epoch}.jpg'

        torch.save(latent_inputs[i], save_filename[:-4]+'.pt')

        image_np = denormalize(fake[i], args.arch_name).data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(save_filename)

    torch.save(latent_inputs, f'{args.final_img_dirname}/latent_inputs.pt')

    # concatenate all best images
    concat_final_images(args.final_img_dirname.rstrip('/'), nrow=args.nrow)
