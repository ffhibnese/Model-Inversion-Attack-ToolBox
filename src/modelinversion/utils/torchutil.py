

import copy
import random
from typing import Callable
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import numpy as np
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.data import sampler

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
            
class BaseHook(metaclass=ABCMeta):
    """Monitor the model when forward
    """
    
    def __init__(self, module: Module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)
        
    @abstractmethod
    def hook_fn(self, module, input, output):
        raise NotImplementedError()
    
    @abstractmethod
    def get_feature(self) -> Tensor:
        """
        Returns:
            Tensor: the value that the hook monitor.
        """
        raise NotImplementedError()
    
    def close(self):
        self.hook.remove()
        
class OutputHook(BaseHook):
    """Monitor the output of the model
    """
    
    def __init__(self, module: Module) -> None:
        super().__init__(module)
        
    def hook_fn(self, module, input, output):
        self.feature = output
        
    def get_feature(self):
        return self.feature
    
    
class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length