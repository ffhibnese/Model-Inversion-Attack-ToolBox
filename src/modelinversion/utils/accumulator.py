from collections import defaultdict

import torch

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0] * n
        self.num = 0

    def add(self, *args, add_num=1, add_type='mean'):
        """adding data to the data list
        """
        assert len(args) == len(self.data)
        mul_coef = add_num if add_type == 'mean' else 1
        self.num += add_num
        for i, add_item in enumerate(args):
            if isinstance(add_item, torch.Tensor):
                add_item = add_item.item()
            self.data[i] += add_item * mul_coef
        

    def reset(self):
        """reset all data to 0
        """
        self.data = [0] * len(self.data)
        self.num = 0

    def __getitem__(self, idx):
        return self.data[idx]
    
    def avg(self, idx = None):
        """Calculate average of the data specified by `idx`. If idx is None, it will calculate average of all data.

        Args:
            idx (int, optional): subscript for the data list. Defaults to None.

        Returns:
            int | list: list if idx is None else int
        """
        num = 1 if self.num == 0 else self.num
        if idx is None:
            return [d / num for d in self.data]
        else:
            return self.data[idx] / num
        
        
class DictAccumulator:
    def __init__(self) -> None:
        self.data = defaultdict(lambda : 0)
        self.num = 0
        
    def reset(self):
        """reset all data to 0
        """
        self.data = defaultdict(lambda : 0)
        self.num = 0
        
    def add(self, add_dic: dict, add_num=1, add_type='mean'):
        mul_coef = add_num if add_type == 'mean' else 1
        self.num += add_num
        for key, val in add_dic.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.data[key] += val * mul_coef
        
    def __getitem__(self, key):
        return self.data[key]
    
    def avg(self, key = None):
        num = 1 if self.num == 0 else self.num
        if key is None:
            # return [d / num for d in self.data]
            return {k: (v / num) for k, v in self.data.items()}
        else:
            return self.data[key] / num