import os
import shutil
from typing import Tuple, Dict, List, Any

from PIL import Image
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import transforms 
from torchvision.transforms import CenterCrop, Resize, Compose
from torchvision.datasets import ImageFolder, DatasetFolder
from tqdm import tqdm

from ..utils.io import IMG_EXTENSIONS
from ..utils import walk_imgs

COPY = 'copy'
MOVE = 'move'
SYMLINK = 'symlink'

def find_classes_folder(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset. 
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    return classes, [os.path.join(directory, c) for c in classes]

def file_transfer(src_path, dst_path, mode = COPY):
    if mode == COPY:
        shutil.copy(src_path, dst_path)
    elif mode == MOVE:
        shutil.move(src_path, dst_path)
    elif mode == SYMLINK:
        os.symlink(src_path, dst_path)
    else:
        raise RuntimeError(f'Invalid mode {mode}')
    
def preprocess_facescrub(src_path, dst_path, mode = COPY, split_seed=42):
    root_actors = os.path.join(src_path, 'actors/faces')
    root_actresses = os.path.join(src_path, 'actresses/faces')
    
    classes_actors, folders_actors = find_classes_folder(root_actors)
    classes_actoresses, folders_actoresses = find_classes_folder(root_actresses)
    classes = classes_actors + classes_actoresses
    folders = folders_actors + folders_actoresses
    
    files = []
    
    for i, folder in enumerate(tqdm(folders, leave=False)):
        # classname = classes[i]
        filenames = sorted([name for name in os.listdir(folder) if name.endswith(IMG_EXTENSIONS)])
        for fname in filenames:
            files.append([i, folder, fname])
        
    
    indices = list(range(len(files)))
    np.random.RandomState(split_seed).shuffle(indices)
    
    training_set_size = int(0.9 * len(indices))
    train_idx = indices[:training_set_size]
    test_idx = indices[training_set_size:]
    
    for split, indices in zip(['train', 'test'], [train_idx, test_idx]):
        root_path = os.path.join(dst_path, split)
        for idx in tqdm(indices, leave=False):
            i, src_folder, fname = files[idx]
            src_img_path = os.path.join(src_folder, fname)
            dst_class_dir = os.path.join(root_path, f'{i}')
            os.makedirs(dst_class_dir, exist_ok=True)
            dst_img_path = os.path.join(dst_class_dir, fname)
            # print(src_img_path)
            # print(dst_img_path)
            # exit()
            file_transfer(src_img_path, dst_img_path, mode=mode)


class _CelebaTransform:
    
    def __init__(self) -> None:
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize((re_size, re_size))
        ])
        
    def trans(self, src_path, dst_path):
        img = Image.open(src_path)
        img = self.transform(img)
        img.save(dst_path)

def split(raw_img_dir, split_file_path, dst_dir, trans):
    with open(split_file_path) as f:
        data = f.readlines()
    for s in tqdm(data, leave=False):
        s = s.strip()
        if s != '':
            s, label = s.split(' ')
            
            src_path = os.path.join(raw_img_dir, f'{s}')
            dst_label_dir = os.path.join(dst_dir, f'{label}')
            os.makedirs(dst_label_dir, exist_ok=True)
            dst_path = os.path.join(dst_label_dir, s[:-3] + 'png')
            
            trans.trans(src_path, dst_path)
            
        
def preprocess_celeba64(src_path, dst_path, split_files_path):
    
    trans = _CelebaTransform()
    
    split_files = [
        'private_train.txt',
        'private_test.txt',
        'public.txt'
    ]
    
    split_files = [os.path.join(split_files_path, filename) for filename in split_files]
    
    dst_dirs = [
        'private/train',
        'private/test',
        'split/public'
    ]
    
    dst_dirs = [os.path.join(dst_path, filename) for filename in split_files]
    
    for dst_dir, split_file_dir in zip(dst_dirs, split_files):
        split(src_path, split_file_dir, dst_dir, trans=trans)
        
def preprocess_celeba224(src_path, dst_path, split_files_path, mode=COPY):
    
    trans = lambda src, dst: file_transfer(src, dst, mode)
    
    split_files = [
        'private_train.txt',
        'private_test.txt',
        'public.txt'
    ]
    
    split_files = [os.path.join(split_files_path, filename) for filename in split_files]
    
    dst_dirs = [
        'private/train',
        'private/test',
        'split/public'
    ]
    
    dst_dirs = [os.path.join(dst_path, filename) for filename in split_files]
    
    for dst_dir, split_file_dir in zip(dst_dirs, split_files):
        split(src_path, split_file_dir, dst_dir, trans)
        
def preprocess_ffhq64(src_path, dst_path):
    src_paths = walk_imgs(src_path)
    
    dst_dir = os.path.join(dst_path, 'images')
    os.makedirs(dst_dir, exist_ok=True)
    
    def to_dst_path(path):
        filename = os.path.split(path)[1]
        return os.path.join(dst_dir, filename)
    dst_paths = list(map(to_dst_path, src_paths))
    
    trans = Compose([
        Image.open,
        CenterCrop((88, 88)),
        Resize((64, 64), antialias=True)
    ])
    
    for src, dst in zip(tqdm(src_paths, leave=False), dst_paths):
        trans_img: Image.Image = trans(src)
        trans_img.save(dst)
        
def preprocess_ffhq256(src_path, dst_path):
    
    src_paths = walk_imgs(src_path)
    
    dst_dir = os.path.join(dst_path, 'images')
    os.makedirs(dst_dir, exist_ok=True)
    
    def to_dst_path(path):
        filename = os.path.split(path)[1]
        return os.path.join(dst_dir, filename)
    dst_paths = list(map(to_dst_path, src_paths))
    
    trans = Compose([
        Image.open,
        CenterCrop((800, 800)),
        Resize((256, 256), antialias=True)
    ])
    
    for src, dst in zip(tqdm(src_paths, leave=False), dst_paths):
        trans_img: Image.Image = trans(src)
        trans_img.save(dst)
        
def preprocess_metfaces256(src_path, dst_path):
    return preprocess_ffhq256(src_path, dst_path)

def preprocess_afhq256(src_path, dst_path):
    
    src_paths = walk_imgs(src_path)
    
    dst_dir = os.path.join(dst_path, 'images')
    os.makedirs(dst_dir, exist_ok=True)
    
    def to_dst_path(path):
        filename = os.path.split(path)[1]
        return os.path.join(dst_dir, filename)
    dst_paths = list(map(to_dst_path, src_paths))
    
    trans = Compose([
        Image.open,
        Resize((256, 256), antialias=True)
    ])
    
    for src, dst in zip(tqdm(src_paths, leave=False), dst_paths):
        trans_img: Image.Image = trans(src)
        trans_img.save(dst)
        
