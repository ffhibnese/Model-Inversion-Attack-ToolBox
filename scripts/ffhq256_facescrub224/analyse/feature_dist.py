import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torchvision.transforms import (
    ToTensor,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    CenterCrop,
    Resize,
    functional as TF,
)

from modelinversion.models import (
    get_stylegan2ada_generator,
    auto_classifier_from_pretrained, TorchvisionClassifierModel
)
from modelinversion.sampler import ImageAugmentSelectLatentsSampler
from modelinversion.utils import (
    augment_images_fn_generator,
    Logger,freeze
)
from modelinversion.attack import (
    IntermediateWhiteboxOptimizationConfig,
    StyelGANIntermediateWhiteboxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
    ImageAugmentClassificationLoss,
)
from modelinversion.datasets import FaceScrub224, ClassSubset
from modelinversion.scores import ImageClassificationAugmentConfidence
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    FaceDistanceMetric,
    ImageFidPRDCMetric,
)



@torch.no_grad()
def main(tag):

    save_name = f'{tag}.pth'
    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'


    experiment_dir = f'./results_feature_dist'
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    kwargs = {}
    if 'neck' in tag:
        kwargs['feature_compressed'] = True
    model = auto_classifier_from_pretrained(
        target_model_ckpt_path, **kwargs
    ).to(device)

    freeze(model)

    eval_dataset_path = (
        '/data/<usrname>/datasets/facescrub/'
    )

    eval_dataset = FaceScrub224(
        eval_dataset_path,
        train=True,
        output_transform=Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    eval_dataset = ClassSubset(
        eval_dataset,
        list(range(100))
    )

    print(len(eval_dataset))

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    all_features = []
    all_labels = []
    from tqdm import tqdm
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        feature = model(images)[1]['feature'].detach().cpu()
        all_features.append(feature)
        all_labels.append(labels)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print('feature shape:', all_features.shape)

    # calculate feature l2 distance between features using torch.cdist
    distance = torch.cdist(all_features, all_features, p=2)

    # create a mask matrix, if i and j are the same class, mask[i][j] = 0, otherwise mask[i][j] = 1
    # mask = torch.zeros_like(distance, dtype=torch.bool)
    mask = all_labels.reshape(1, -1) != all_labels.reshape(-1, 1)
    # for i in range(len(distance)):
    #     for j in range(len(distance)):
    #         if all_labels[i] == all_labels[j]:
    #             mask[i][j] = 0
    #         else:
    #             mask[i][j] = 1
    # mask = mask.reshape(-1)
    # distance = distance.reshape(-1)

    # inter_dist = distance[mask]
    # intra_dist = distance[~mask]
    mask = mask.float()
    inter_dist_avg = (distance * mask).sum(dim=-1) / mask.sum(dim=-1)
    intra_dist_avg = (distance * (1-mask)).sum(dim=-1) / (1-mask).sum(dim=-1)

    distance_mul_mask = (distance * mask)
    distance_mul_not_mask = distance * (1-mask)

    distance_mul_mask[distance_mul_mask == 0] = 99999999
    distance_mul_not_mask[distance_mul_not_mask == 0] = 99999999

    inter_dist_min = distance_mul_mask.min(dim=-1)[0]
    intra_dist_min = distance_mul_not_mask.min(dim=-1)[0]



    
    os.makedirs(experiment_dir, exist_ok=True)

    torch.save({
        'inter_dist_avg': inter_dist_avg,
        'intra_dist_avg': intra_dist_avg,
        'inter_dist_min': inter_dist_min,
        'intra_dist_min': intra_dist_min,
    }, os.path.join(experiment_dir, save_name))


        

if __name__ == '__main__':
    device_ids_available = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available

    for tag in ['no', 'neck50tanh','bido_ih0.15_oh1.5','ls0.01','tl0.7','vib0.005']:
        main(tag)
