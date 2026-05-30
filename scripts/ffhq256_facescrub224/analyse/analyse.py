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
def main(tag, feature_compressed=False):

    save_name = f'{tag}.pth'
    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'


    experiment_dir = f'./results_tsne'
    target_model_ckpt_path = f'../result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    model = auto_classifier_from_pretrained(
        target_model_ckpt_path
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
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Example data: 100 samples with 50 features
    np.random.seed(42)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    os.makedirs(experiment_dir, exist_ok=True)

    torch.save({
        'features': all_features,
        'labels': all_labels,
        'features_2d': features_2d,
    }, os.path.join(experiment_dir, save_name))


        

if __name__ == '__main__':
    device_ids_available = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_available

    for tag in ['no', 'neck50tanh','bido_ih0.15_oh1.5','ls0.01','tl0.7','vib0.005']:
        main(tag)
