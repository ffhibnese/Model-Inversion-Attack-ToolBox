
import torch

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0] * n
        self.num = 0

    def add(self, *args):
        # self.data = [a + float(b) for a, b in zip(self.data, args)]
        assert len(args) == len(self.data)
        self.num += 1
        for i, add_item in enumerate(args):
            if isinstance(add_item, torch.Tensor):
                add_item = add_item.item()
            self.data[i] += add_item
        

    def reset(self):
        self.data = [0] * len(self.data)
        self.num = 0

    def __getitem__(self, idx):
        return self.data[idx]
    
    def avg(self, idx = None):
        num = 1 if self.num == 0 else self.num
        if idx is None:
            return [d / num for d in self.data]
        else:
            return self.data[idx] / num