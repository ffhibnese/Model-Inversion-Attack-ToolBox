from tqdm import tqdm
from typing import Optional

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from ..models.base import ModelMixin
from ..utils import unwrapped_parallel_module

def _get_first(data):
    if not isinstance(data, Tensor):
        return data[0]
    return data

def train_mapping_model(
    epoch_num: int,
    mapping_module: ModelMixin,
    optimizer: Optimizer,
    src_model: Module,
    dst_model: Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    schedular: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    show_info_iters: int = 100,
):
    src_model.eval()
    dst_model.eval()

    loss_fn = nn.MSELoss()

    for epoch in range(epoch_num):

        bar = tqdm(dataloader, leave=False)
        for i, data in enumerate(bar):
            data = _get_first(data).to(device)
            with torch.no_grad():
                inputs = _get_first(src_model(data)).softmax(dim=-1)
                labels = _get_first(dst_model(data))

            map_result = mapping_module(inputs)
            loss = loss_fn(map_result, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % show_info_iters == 0:
                bar.set_description_str(f'epoch: {epoch} loss: {loss.item():.5f}')

        if schedular is not None:
            schedular.step()

        unwrapped_parallel_module(mapping_module).save_pretrained(save_path)