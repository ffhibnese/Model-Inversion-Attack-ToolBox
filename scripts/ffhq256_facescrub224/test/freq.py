import sys
import os
import argparse
import time

sys.path.append('../../../src')

import torch
from torch import nn
from modelinversion.datasets import FaceScrub224, CelebA224
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from kornia import augmentation

from modelinversion.models import (
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained, TorchvisionClassifierModel
)
from modelinversion.sampler import SimpleLatentsSampler
from modelinversion.utils import (
    unwrapped_parallel_module,
    augment_images_fn_generator,
    Logger,freeze, DictAccumulator
)
from modelinversion.attack import (
    ImageAugmentWhiteBoxOptimizationConfig,
    ImageAugmentWhiteBoxOptimization,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from modelinversion.metrics import (
    ImageClassifierAttackAccuracy,
    ImageDistanceMetric,
    ImageFidPRDCMetric,
    FaceDistanceMetric,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_low_filter_mask(image, cutoff_frequency):
    C, H, W = image.shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_y, center_x = H // 2, W // 2
    distance = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    filter_mask = distance <= cutoff_frequency
    return filter_mask

def pass_filter(image, filter_mask):
    # 假设输入图像为PyTorch张量，且形状为 (C, H, W)
    C, H, W = image.shape
    # 对图像进行二维傅里叶变换
    image_fft = torch.fft.fft2(image)
    image_fft_shifted = torch.fft.fftshift(image_fft)
    
    # 应用低通滤波器
    image_fft_shifted_filtered = image_fft_shifted * filter_mask
    
    # 逆傅里叶变换回到空间域
    image_fft_filtered = torch.fft.ifftshift(image_fft_shifted_filtered)
    image_filtered = torch.fft.ifft2(image_fft_filtered)
    
    return torch.abs(image_filtered)

@torch.no_grad()
def main():

    tag = ''
    target_model_ckpt_path = f'../result_classifier/train_facescrub64_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    eval_model_ckpt_path_2 = '/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/facescrub224/facescrub224_inception_v3_94.45.pth'
    eval_model_ckpt_path = '/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub64_maxvit_t/facescrub224_maxvit_t.pth'

    eval_dataset_path = (
        '/data/<usrname>/datasets/facescrub/'
    )

    celeba_dataset_path = (
        '/data/<usrname>/datasets/pre_celeba_high/private_train'
    )

    pri_dataset = FaceScrub224(
        eval_dataset_path,
        train=True,
        output_transform=Compose([
            ToTensor()
        ])
    )

    celeba_dataset = CelebA224(
        celeba_dataset_path,
        output_transform=Compose([
            ToTensor()
        ])
    )

    datas = {
        'pri': pri_dataset,
        'celeba': celeba_dataset
    }

    device = torch.device('cuda')

    model1 = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
    model2 = auto_classifier_from_pretrained(eval_model_ckpt_path).to(device)
    # model2 = TorchvisionClassifierModel('inception_v3', num_classes=530, resolution=299, operate_aux=False)
    # model2.load_state_dict(torch.load(eval_model_ckpt_path_2, map_location='cpu')['state_dict'])
    # model2.to(device)

    # model2.save_pretrained(eval_model_ckpt_path_2)

    model3 = auto_classifier_from_pretrained(eval_model_ckpt_path).to(device)

    model1.eval()
    model2.eval()
    model3.eval()

    freeze(model1)
    freeze(model2)
    freeze(model3)

    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    image, label = pri_dataset[0]
    threshold = list(range(5, 100))
    high_freq_images = []
    low_freq_images = []
    from torchvision.utils import save_image
    save_image(image, './images/ori.png')
    for th in threshold:
        low_mask = get_low_filter_mask(image, th)
        high_mask = ~low_mask
        high_freq_images.append(pass_filter(image, high_mask))
        low_freq_images.append(pass_filter(image, low_mask))

        save_image(high_freq_images[-1], f'./images/high_{th}.png')
        save_image(low_freq_images[-1], f'./images/low_{th}.png')

    high_freq_images = normalize(torch.stack(high_freq_images)).to(device)
    low_freq_images = normalize(torch.stack(low_freq_images)).to(device)
    images = normalize(image).unsqueeze(0).to(device)

    for i, model in enumerate([model1, model2, model3], start=1):

        confs_ori = model(images)[0].softmax(dim=-1)
        confs_low = model(low_freq_images)[0].softmax(dim=-1)
        confs_high = model(high_freq_images)[0].softmax(dim=-1)

        low_consine_similarity = torch.cosine_similarity(confs_ori, confs_low, dim=-1)
        high_consine_similarity = torch.cosine_similarity(confs_ori, confs_high, dim=-1)

        pseudo_label = torch.argmax(confs_ori, dim=-1).item()
        low_conf = confs_low[:, pseudo_label]
        high_conf = confs_high[:, pseudo_label]
        torch.save({
                'low_conf': low_conf,
                'high_conf': high_conf,
                'low_consine_similarity': low_consine_similarity,
                'high_consine_similarity': high_consine_similarity
            }, f'./freq{i}.pt'
        )
    



    # for dataset_name, dataset in datas.items():
    #     print(dataset_name)
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=300,
    #         num_workers=8,
    #         shuffle=False,
    #     )

    #     accumulator = DictAccumulator()

    #     def calc_cos_similarity(logits1, logits2):
    #         return torch.cosine_similarity(logits1, logits2, dim=-1).mean().item()

    #     from tqdm import tqdm
    #     for images, labels in tqdm(dataloader, leave=False):
    #         images = images.to(device)
    #         labels = labels.to(device)

    #         logits1 = model1(images)[0].softmax(dim=-1)
    #         logits2 = model2(images)[0].softmax(dim=-1)
    #         logits3 = model3(images)[0].softmax(dim=-1)

    #         accumulator.add({
    #             'cos_similarity': calc_cos_similarity(logits1, logits2),
    #             'cos_similarity_2': calc_cos_similarity(logits1, logits3),
    #             'cos_similarity_3': calc_cos_similarity(logits2, logits3)
    #         }, len(images))

    #     print(accumulator.avg())
            

if __name__ == '__main__':
    main()