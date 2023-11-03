from .basemetric import BaseMetricCalculator
import torch
from ..utils import Record

class AccCalculator(BaseMetricCalculator):
    
    def __init__(self, model, recover_imgs_dir, real_imgs_dir, batch_size=60, label_spec=False, device='cpu') -> None:
        super().__init__(model, recover_imgs_dir, real_imgs_dir, batch_size, label_spec, device)
        
    def calculate(self):
        dataloader = self.get_dataloader(self.recover_imgs_dir)
        
        record = Record(4)
        
        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)
            pred = self.model(imgs).result.detach().cpu()
            
            # batch, 10
            top10 = torch.topk(pred, 10, dim=-1).cpu()
            eq = (top10 == pred.unsqueeze(-1)).sum(dim=0)
            bs = len(labels)
            
            record.add(eq[0], eq[:5].sum(), eq[:10].sum(), bs)
            
        if record[-1] == 0:
            raise RuntimeError("no img")
            
        acc, acc5, acc10 = record[0] / record[-1], record[1] / record[-1], record[2] / record[-1]
            
        return acc, acc5, acc10