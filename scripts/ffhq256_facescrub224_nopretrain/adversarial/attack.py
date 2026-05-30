import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torchattacks
from torch import nn
import sys

sys.path.append('/mnt/data/<usrname>/mywork/lora_defense/src')
from modelinversion.models import auto_classifier_from_pretrained
from modelinversion.datasets import FaceScrub224, ClassSubset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from modelinversion.utils import set_random_seed, Logger


class ModelWrapper(nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


eval_dataset_path = '/mnt/data/<usrname>/datasets/facescrub/'

eval_dataset = FaceScrub224(
    eval_dataset_path,
    train=False,
    output_transform=Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
)

eval_dataset = ClassSubset(eval_dataset, list(range(100)))

import torch
from torch.utils.data import DataLoader
import torchattacks.attack
from tqdm import tqdm

eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False, num_workers=16)


def attack(tag, atk: torchattacks.attack.Attack, target=False, device='cuda'):

    set_random_seed(42)
    target_s = 'target' if target else 'untarget'

    total_nums = 0
    # clean_corrects = 0
    adv_corrects = 0
    atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if target:
        atk.set_mode_targeted_by_label(quiet=True)

    for i, (x, y) in enumerate(tqdm(eval_loader, leave=False)):
        x = x.to(device)
        y = y.to(device)
        target_y = y
        if target:
            target_y = (y + 1) % 530
        adv_x = atk(x, target_y)

        with torch.no_grad():
            adv_y = model(adv_x)
            adv_y = torch.argmax(adv_y, dim=1)
            adv_corrects += (adv_y == target_y).sum().item()

            # clean_y = model(x)
            # clean_y = torch.argmax(clean_y, dim=1)
            # clean_corrects += (clean_y == y).sum().item()

        total_nums += x.shape[0]

    # print(adv_x.shape, adv_x.device, total_nums, clean_corrects, adv_corrects)
    # break

    if not target:
        adv_corrects = total_nums - adv_corrects
    # clean_acc, asr = clean_corrects / total_nums, adv_corrects / total_nums
    asr = adv_corrects / total_nums
    # print(clean_acc, asr)
    print(f'{tag}_{atk.__class__.__name__}_{target_s}, asr: {asr}')
    # logger.close()
    return asr


import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.makedirs('./results', exist_ok=True)


logger = Logger('./results_test', f'atk.log')

all_results = []

# for tag in ['no', 'ftneck32tanh','bido_ih0.05_oh0.5','ls0.02_scheduler','tl0.6','vib0.005', 'trap0.05_0.2']:
for tag in ['rolss0.0_1']:
# for tag in ['ftneck32tanh']:

    tag_result = [tag]

    _tag = tag

    if tag == 'no':
        tag = ''
    else:
        tag = f'_{tag}'

    target_model_ckpt_path = f'/mnt/data/<usrname>/mywork/lora_defense/test_lora/ffhq256_facescrub224_nopretrain/result_classifier/train_facescrub224_resnet152{tag}/facescrub224_resnet152{tag}.pth'

    model = auto_classifier_from_pretrained(target_model_ckpt_path)
    model = ModelWrapper(model).to('cuda')
    model.eval()


    for target in [False, True]:
        atk = torchattacks.FGSM(model, eps=2 / 255)
        res = attack(_tag, atk, target=target)
        tag_result.append(res)

        atk = torchattacks.PGD(
            model, eps=2 / 255, alpha=2 / 255, steps=10, random_start=True
        )
        res = attack(_tag, atk, target=target)
        tag_result.append(res)

        atk = torchattacks.BIM(model, eps=2 / 255, alpha=2 / 255, steps=10)
        res = attack(_tag, atk, target=target)
        tag_result.append(res)

        atk = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10)
        res = attack(_tag, atk, target=target)
        tag_result.append(res)

    all_results.append(tag_result)

torch.save(all_results, './results_test3/all_results2.pth')

import pandas as pd

df = pd.DataFrame(all_results)
df.to_csv('./results_test3/all_results2.csv', index=False)
