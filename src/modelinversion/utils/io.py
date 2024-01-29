import os
import yaml
from collections import OrderedDict
import torch

def safe_save(obj, save_dir, save_name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, os.path.join(save_dir, save_name))
    
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    
def walk_imgs(path):
    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(IMG_EXTENSIONS):
                img_paths.append(os.path.join(root, file))
    return img_paths

yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
yaml.add_representer(tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data))

def print_as_yaml(obj, stdout=True, file=None, mode='w'):
    s = yaml.dump(obj)
    
    if stdout:
        print(s)
    if file:
        with open(file, mode) as f:
            f.write(s)
    