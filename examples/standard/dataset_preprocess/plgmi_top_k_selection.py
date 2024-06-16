import sys
import os

sys.path.append('../../../src')

import torch
from torch import nn
from torchvision.transforms import functional as TF

from modelinversion.models import IR152_64
from modelinversion.datasets import top_k_selection

if __name__ == '__main__':

    top_k = 30
    num_classes = 1000
    target_model_ckpt_path = '<fill it>'
    src_dataset_path = '<fill it>'
    dst_dataset_path = '<fill it>'

    batch_size = 50
    device_ids_available = '0'

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    target_model = IR152_64(num_classes=num_classes)
    target_model.load_state_dict(
        torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    )
    target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)

    # dataset generation

    top_k_selection(
        top_k=top_k,
        src_dataset_path=src_dataset_path,
        dst_dataset_path=dst_dataset_path,
        batch_size=batch_size,
        target_model=target_model,
        num_classes=num_classes,
        device=device,
        create_aug_images_fn=lambda img: TF.resize(img, (64, 64), antialias=True),
    )
