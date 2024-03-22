#!/usr/bin/env python3
# coding=utf-8
import torch
from torch import nn
import torchvision.models as models

# import vgg_m_face_bn_dag
from .....models import *
# import net_sphere
import os

from collections import OrderedDict
def get_model(arch_name, device, classifier_dir, dataset_name, use_dropout=False):
    
    path = os.path.join(classifier_dir, 'target_eval', dataset_name)
    # path = None
    # state_dict = torch
    if arch_name == 'vgg16':
        path = os.path.join(path, 'VGG16_88.26.tar')
        model  = (VGG16(1000))
    elif arch_name == 'ir152':
        path = os.path.join(path, 'IR152_91.16.tar')
        model = (IR152(1000))
    elif arch_name == 'facenet64':
        path = os.path.join(path, 'FaceNet64_88.50.tar')
        model = (FaceNet64(1000))
    elif arch_name == 'facenet':
        path = os.path.join(path, 'FaceNet_95.88.tar')
        model = (FaceNet(1000))
    elif arch_name =='resnet50_scratch_dag':
        path = os.path.join(path, 'resnet50_scratch_dag.pth')
        model = Resnet50_scratch_dag()
    elif arch_name == 'vgg_face_dag':
        path = os.path.join(path, 'vgg_face_dag.pth')
        model = Vgg_face_dag(use_dropout=use_dropout)
    elif arch_name == 'inception_resnetv1_vggface2':
        path = os.path.join(path, '20180402-114759-vggface2.pt')
        # model = InceptionResnetV1(classify=True, pretrained='vggface2', ckpt_path=path)
        model = InceptionResnetV1(classify=True, pretrained=None, ckpt_path=path, num_classes=8631)
    # elif arch_name == 'inception_resnetv1_casia':
    #     model = InceptionResnetV1(classify=True, pretrained='casia-webface')
    else:
        raise RuntimeError('arch name error')
    # print(f'>>>>>>>>> arch name: {arch_name}  path {path}')
    
    if path is not None and os.path.isfile(path):
    
        state_dict = torch.load(path, map_location=device)
        
        if isinstance(state_dict, dict) and 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        
    
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k
        #     if k.startswith('module.'):
        #         pl = nn.DataParallel(model)
        #         pl.load_state_dict(state_dict)
        #         torch.save({'state_dict': pl.module.state_dict()}, path)
        #         state_dict = torch.load(path)['state_dict']
        #         break
        model.load_state_dict(state_dict)
    else:
        print(path)
        raise RuntimeError('not checkpoint')
    return model.to(device)