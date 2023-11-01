# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------
from dataclasses import dataclass

@dataclass
class DeepInversionArgs:
    
    # exp_name: str
    adi_scale: float
    device: str
    bs: int
    lr: float
    target_name: str
    eval_name: str
    do_flip: bool
    r_feature: float
    target_labels: list
    # save_dir: str
    
    worldsize = 1
    local_rank = 0
    tv_l1 = 0.
    tv_l2 = 0.0001
    l2 = 0.00001
    main_loss_multiplier = 1.
    
    store_best_images = True
    epochs = 20000
    setting_id = 0
    first_bn_multiplier = 10
    
    # fp16 = False
    jitter = 30
    comment = ''

from dataclasses import dataclass
import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms
from .deepinversion import DeepInversionClass

import numpy as np
import torch.cuda.amp as amp
import os
import torchvision.models as models
from .utils.utils import load_model_pytorch, distributed_is_initialized
from ...utils import Tee

random.seed(0)


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def deepinversion_attack(args: DeepInversionArgs, target_model, eval_model, folder_manager):
    torch.manual_seed(args.local_rank)
    device = torch.device(args.device)

    # if args.target_name == "resnet50v15":
    #     from .models.resnetv15 import build_resnet
    #     net = build_resnet("resnet50", "classic")
    # else:
    #     print("loading torchvision model for inversion with the name: {}".format(args.target_name))
    #     net = models.__dict__[args.target_name](pretrained=True)

    # print(f'device: {device}')
    # net = net.to(device)

    print('==> Resuming from checkpoint..')

    target_model.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.eval_name is not None and args.adi_scale == 0:
        # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
        if args.local_rank == 0:
            print("loading verifier: ", args.eval_name)
            net_verifier = models.__dict__[args.eval_name](pretrained=True).to(device)
            net_verifier.eval()

    if args.adi_scale != 0.0:
        student_arch = "resnet18"
        net_verifier = models.__dict__[student_arch](pretrained=True).to(device)
        net_verifier.eval()

        net_verifier = net_verifier.to(device)
        net_verifier.train()

    # final images will be stored here:
    # adi_data_path =  os.path.join(exp_name, 'final_images') #"./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    # exp_name = os.path.join(exp_name, 'generations') # f"{exp_name}/generations/"

    args.iterations = 2000
    args.start_noise = True
    # args.detach_student = False

    args.resolution = 224
    bs = args.bs
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = args.do_flip
    # parameters["random_label"] = args.random_label
    parameters["store_best_images"] = args.store_best_images

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.eval_name is not None:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    # assert not (use_fp16 and device == 'cpu'), 'cpu do not support fp16'
    DeepInversionEngine = DeepInversionClass(target_labels=args.target_labels,
                                             net_teacher=target_model,
                                             folder_manager=folder_manager,
                                            #  final_data_path=args.save_dir,
                                            #  path=exp_name,
                                             parameters=parameters,
                                             setting_id=args.setting_id,
                                             bs = bs,
                                            #  use_fp16 = args.fp16,
                                             jitter = jitter,
                                             criterion=criterion,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function,
                                             hook_for_display = hook_for_display,
                                             device=device)
    net_student=None
    if args.adi_scale != 0:
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student)
    
