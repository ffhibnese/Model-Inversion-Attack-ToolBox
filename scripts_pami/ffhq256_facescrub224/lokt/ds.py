import sys
import os

sys.path.append('../../../src')

import torch
from torch import nn
import torchvision.transforms as TF

from modelinversion.models import (
    LoktGenerator256,
    IR152_64,
    auto_classifier_from_pretrained,
    auto_generator_from_pretrained,
)
from modelinversion.datasets import (
    generator_generate_datasets,
    preprocess_facescrub_fn,
    GeneratorDataset,
)


def main(tag):

    if tag == 'no':
        tag = ''
    else:
        tag = '_' + tag

    target_model_ckpt_path = f'/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'
    # dataset_path = '/mnt/data/<usrname>/datasets/ffhq64'

    # if tag == 'no':
    #     target_model_ckpt_path = f'../../../checkpoints_v2/classifier/facescrub64/facescrub64_ir152_98.25.pth'
    # elif tag == 'tl0.5':
    #     target_model_ckpt_path = f'/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/facescrub64/facescrub64_ir152_tl_0.5_95.36.pth'
    # else:
    #     target_model_ckpt_path = f'../result_classifier/train_facescrub64_ir152_{tag}/facescrub64_ir152_{tag}.pth'

    # if tag == 'no':
    #     tag = ''
    # else:
    tag = '_' + tag

    num_classes = 530
    generator_ckpt_path = f'./gan/lokt_ffhq256_facescrub224_ir152{tag}_gan/G.pth'
    dst_dataset_path = (
        f'./dataset/lokt_ffhq256_facescrub224_ir152{tag}_dataset/dataset.pt'
    )

    batch_size = 200
    device_ids_str = '1'

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target models

    # dataset generator

    z_dim = 128

    # generator = LoktGenerator64(num_classes, dim_z=z_dim)
    # state_dict = torch.load(generator_ckpt_path, map_location='cpu')['state_dict']

    # for k in list(state_dict.keys()):
    #     newk = None
    #     if 'l1' in k:
    #         newk = k.replace(
    #             'l1',
    #             'block1.0',
    #         )
    #     elif 'conv6' in k:
    #         newk = k.replace(
    #             'conv6',
    #             'block6.0',
    #         )
    #     elif 'b6' in k:
    #         newk = k.replace(
    #             'b6',
    #             'block5_res.0',
    #         )
    #     if newk is not None:
    #         state_dict[newk] = state_dict[k]
    # del state_dict[k]

    # generator.load_state_dict(state_dict)
    generator = auto_generator_from_pretrained(generator_ckpt_path)
    generator = generator.to(device)
    generator.eval()

    # prepare target models

    # target_model = IR152_64(num_classes=num_classes)
    # target_model.load_state_dict(
    #     torch.load(target_model_ckpt_path, map_location='cpu')['state_dict']
    # )
    # target_model = nn.DataParallel(target_model, device_ids=gpu_devices).to(device)
    target_model = auto_classifier_from_pretrained(target_model_ckpt_path).to(device)
    target_model.eval()

    # generator_generate_datasets(
    #     dst_dataset_path,
    #     generator,
    #     num_per_class=50,
    #     num_classes=num_classes,
    #     batch_size=batch_size,
    #     input_shape=z_dim,
    #     target_model=target_model,
    #     device=device,
    # )

    dataset = GeneratorDataset.create(
        z_dim,
        num_classes=num_classes,
        generate_num_per_class=500,
        generator=generator,
        target_model=target_model,
        batch_size=batch_size,
        device=device,
    )

    dataset.save(dst_dataset_path)


# main('no')
# tags = [
#     'no',
#     'neck30tanh_focal8_lr1_5',
#     'tl0.5',
#     'bido0.01_0.1',
#     'ls_-0.3',
#     'vib0.1',
#     'rolss0.0_2',
# ]
for tag in [
    'no',
    # 'vib0.01',
    # 'bido0.01_0.1_pretrain',
    # 'ls0.05',
    # 'tl0.5',
    # 'rolss0.0_2',
]:
    #     main(tag, 6)

    # # main(tags[1])

    # for tag in tags:
    try:
        main(tag)
    except Exception as e:
        pass
    # exit()

# for tag in tags[:-1]:
#     if os.fork() == 0:
#         main(tag)
#         exit()
#     # main(tag)

# main(tag[-1])
