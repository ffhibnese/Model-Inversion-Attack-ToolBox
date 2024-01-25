import os
import torch

def safe_save(obj, save_dir, save_name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, os.path.join(save_dir, save_name))