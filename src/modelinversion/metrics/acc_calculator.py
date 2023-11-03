from .basemetric import BaseMetricCalculator
import torch
from ..utils import Record

class AccCalculator(BaseMetricCalculator):
    
    def __init__(self, model, recover_imgs_dir, real_imgs_dir=None, recover_feat_dir=None, real_feat_dir=None, batch_size=60, device='cpu') -> None:
        super().__init__(model, recover_imgs_dir, real_imgs_dir, recover_feat_dir, real_feat_dir, batch_size, False, device)
        
    def calculate(self):
        dataloader = self.get_dataloader(self.recover_imgs_dir)
        
        record = Record(4)
        
        self.model.eval()
        with torch.no_grad():
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
            
            
        print(f'acc: {acc: .6f}\t acc5: {acc5:.6f}\t acc10: {acc10:.6f}')
        return acc, acc5, acc10