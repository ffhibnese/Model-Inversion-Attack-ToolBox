
from typing import Callable
from typing import Optional

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
            traverse_module(child, fn)
            
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
            

    
    

    
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
        
def unwrapped_parallel_module(module):
    
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    return module

def augment_images_fn_generator(initial_transform: Optional[Callable]=None,  add_origin_image=True, augment: Optional[Callable]=None, augment_times: int = 0):
    
    def fn(image):
        if initial_transform is not None:
            image = initial_transform(image)
            
        if add_origin_image:
            yield image
            
        if augment is not None:
            for i in range(augment_times):
                yield augment(image)