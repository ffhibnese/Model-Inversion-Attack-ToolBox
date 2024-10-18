from typing import Callable
from typing import Optional

import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def traverse_module(module: nn.Module, fn: Callable, call_middle=False):
    """Use DFS to traverse the module and visit submodules by function `fn`.

    Args:
        module (nn.Module): the module to be traversed
        fn (Callable): visit function
        call_middle (bool, optional): If true, it will visit both intermediate nodes and leaf nodes, else, it will only visit leaf nodes. Defaults to False.
    """

    children = list(module.children())
    if len(children) == 0:
        fn(module)
    else:
        if call_middle:
            fn(module)
        for child in children:
            traverse_module(child, fn, call_middle=call_middle)


def _traverse_name_module_impl(module_tuple: list, fn: Callable, call_middle=False):
    name, module = module_tuple
    children = list(module.named_children())
    if len(children) == 0:
        fn(module_tuple)
    else:
        if call_middle:
            fn(module_tuple)
        for child in children:
            _traverse_name_module_impl(child, fn)


def traverse_name_module(module: nn.Module, fn: Callable, call_middle=False):
    """Use DFS to traverse the module and visit submodules by function `fn`.

    Args:
        module (nn.Module): the module to be traversed
        fn (Callable): visit function
        call_middle (bool, optional): If true, it will visit both intermediate nodes and leaf nodes, else, it will only visit leaf nodes. Defaults to False.
    """
    children = list(module.named_children())
    for child in children:
        _traverse_name_module_impl(child, fn, call_middle=call_middle)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)


def freeze_front_layers(module, ratio=0.5):

    if ratio < 0 or ratio > 1:
        raise RuntimeError('Ratio should be in [0, 1]')

    if ratio == 0:
        unfreeze(module)
        return

    if ratio == 1:
        freeze(module)
        return

    all_modules = []

    def _visit_fn(module):
        all_modules.append(module)

    traverse_module(module, _visit_fn)
    length = len(all_modules)
    if length == 0:
        return

    freeze_line = ratio * length
    for i, m in enumerate(all_modules):
        if i < freeze_line:
            m.requires_grad_(False)
        else:
            m.requires_grad_(True)


def unwrapped_parallel_module(module):

    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    return module


def reparameterize(mu, std):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """

    # std = torch.exp(0.5 * std)
    eps = torch.randn_like(std)

    return eps * std + mu


def augment_images_fn_generator(
    initial_transform: Optional[Callable] = None,
    add_origin_image=True,
    augment: Optional[Callable] = None,
    augment_times: int = 0,
):
    """Return a function for image augmentation.

    Args:
        initial_transform (Optional[Callable], optional): The first transformation to perform. Defaults to None.
        add_origin_image (bool, optional): Whether to return the original image. Defaults to True.
        augment (Optional[Callable], optional): The augmentation to perform. Defaults to None.
        augment_times (int, optional): Times for augmentation to repeat. Defaults to 0.
    """

    def fn(image):
        if initial_transform is not None:
            image = initial_transform(image)

        if add_origin_image:
            yield image

        if augment is not None:
            for i in range(augment_times):
                yield augment(image)

    return fn
