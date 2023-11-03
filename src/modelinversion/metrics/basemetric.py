
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tv_transforms
import os
import torch
import collections
import numpy as np

IMG_SUFFIX = (
    'jpg',
    'jpeg',
    'png',
    'bmp'
)

class BaseMetricCalculator:
    
    def __init__(self, model, recover_imgs_dir, real_imgs_dir, batch_size = 60, label_spec = True, device='cpu') -> None:
        self.recover_imgs_dir = recover_imgs_dir
        self.real_imgs_dir = real_imgs_dir
        self.device = device
        self.model = model.to(self.device)
        self.label_spec = label_spec
        self.batch_size = batch_size
        
    def generate_feature(self, save_dir, dataloader):
        results = collections.defaultdict(list)
        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)
            feature = self.model(imgs).feat[-1]
            label = labels[0].item()
            results[label].append(feature.detach().cpu().numpy())
            
        for label, feats in results.items():
            feats = np.concatenate(feats, axis=0)
            np.save(os.path.join(save_dir, f'{label}.npy'), feats)
            
    def calculate(self):
        raise NotImplementedError()
    
    def get_recover_loader(self):
        return self.get_dataloader(self.recover_imgs_dir)
    
    def get_real_loader(self):
        return self.get_dataloader(self.recover_imgs_dir)
    
    def get_dataloader(self, imgs_dir):
        if self.label_spec:
            def label_spec_loader(imgs_dir):
                raw_labels: list[str] = os.listdir(imgs_dir)
                labels = [label for label in raw_labels if label.isdigit() and os.path.isdir(os.path.join(imgs_dir, label))]
                trans = tv_transforms.ToTensor()
                
                # num_labels = len(labels)
                while True:
                    for label in labels:
                        label_dir = os.path.join(imgs_dir)
                        img_names = [img_name for img_name in os.listdir(label_dir) if img_name.endswith(IMG_SUFFIX)]
                        if len(img_names) == 0:
                            continue
                        
                        imgs = []
                        for img_name in img_names:
                            img_path = os.path.join(label_dir, img_name)
                            imgs.append(trans(Image.open(img_path)))
                            
                        i = 0
                        num_label_imgs = len(imgs)
                        for i in range((num_label_imgs - 1) // self.batch_size + 1):
                            indices = slice(i * self.batch_size, min((i+1)*self.batch_size, num_label_imgs))
                            select_imgs = imgs[indices]
                            res_tensors = torch.stack(select_imgs, dim=0).to(self.device)
                            res_labels = torch.ones((len(select_imgs),), dtype=torch.long, device=self.device)
                            yield res_tensors, res_labels
                
            return label_spec_loader(imgs_dir)
        
        else:
            dataset = ImageFolder(imgs_dir, transform=tv_transforms.ToTensor())
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)