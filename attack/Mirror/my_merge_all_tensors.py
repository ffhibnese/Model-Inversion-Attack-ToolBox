#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import glob

import torch


def merge(root_dir, remove=False):
    files = sorted(glob.glob(os.path.join(root_dir, 'sample_*_img_logits.pt')))
    print('#files,', len(files))
    all_res = []
    for f in files:
        all_res.append(torch.load(f, map_location=torch.device('cpu')))
    all_res = torch.cat(all_res, dim=0)
    print('all_res.shape', all_res.shape)
    torch.save(all_res, os.path.join(root_dir, 'all_logits.pt'))

    if remove:
        for f in files:
            os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true', help='remove files used to merge')
    parser.add_argument('root_dir')
    args = parser.parse_args()
    merge(args.root_dir, remove=args.remove)
