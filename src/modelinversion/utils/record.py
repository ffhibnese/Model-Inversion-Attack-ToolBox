import torch

class Records:
    
    def __init__(self, num) -> None:
        self.records = [0 for _ in range(num)]
        
    def add(self, ls):
        for i, src in enumerate(ls):
            if isinstance(src, torch.Tensor):
                src = src.detach().cpu().numpy()
            self.records[i] += src
            
    def __getitem__(self, index):
        return self.records[index]