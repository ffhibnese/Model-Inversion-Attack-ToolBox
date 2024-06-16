from tqdm import tqdm
from torch import Tensor
from .base import *


@torch.no_grad()
def generate_feature_statics(dataloader, sample_num, classifier, device):

    features = []
    for imgs in tqdm(dataloader, leave=False):
        if not isinstance(imgs, Tensor):
            imgs = imgs[0]

        if sample_num <= 0:
            break
        if sample_num < len(imgs):
            imgs = imgs[sample_num:]
        sample_num -= len(imgs)

        imgs = imgs.to(device)
        _, addition_info = classifier(imgs)
        if not HOOK_NAME_FEATURE in addition_info:
            raise RuntimeError(
                f'{HOOK_NAME_FEATURE} are not contains in the output of the classifier'
            )
        features.append(addition_info[HOOK_NAME_FEATURE].cpu())
    features = torch.cat(features, dim=0)
    features_mean = torch.mean(features, dim=0)
    features_std = torch.std(features, dim=0)
    return features_mean, features_std
