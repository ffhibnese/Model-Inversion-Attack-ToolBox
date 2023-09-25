#!/usr/bin/env python3
# coding=utf-8
import os

import torch
import torch.nn as nn

from .model import Classifier


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def get_classifier(device, grayscale):
    if grayscale:
        print('use grayscale classifier')
        classifier = Classifier(nc=1, ndf=128, nz=530)
        checkpoint = torch.load(os.path.join(THIS_DIR, 'classifier.pth'))
    else:
        print('use rgb classifier')
        classifier = Classifier(nc=3, ndf=128, nz=530)
        checkpoint = torch.load(os.path.join(THIS_DIR, 'classifier_rgb.pth'))
    weight = checkpoint['model']
    _prefix = 'module.'
    weight = {k[len(_prefix):]:v for k, v in weight.items()}
    classifier.load_state_dict(weight)
    return classifier.to(device)
