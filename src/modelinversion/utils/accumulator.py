
import torch

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0] * n
        self.num = 0

    def add(self, *args):
        """adding data to the data list
        """
        assert len(args) == len(self.data)
        self.num += 1
        for i, add_item in enumerate(args):
            if isinstance(add_item, torch.Tensor):
                add_item = add_item.item()
            self.data[i] += add_item
        

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