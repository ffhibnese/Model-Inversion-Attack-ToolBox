import sys
import os
import time

sys.path.append('../../../src')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    Resize,
)

from modelinversion.models import (
    IR152_64,
    VibWrapper,
    BiDOWrapper,
    SkipConnectionWrapper,
    auto_classifier_from_pretrained,
)
from modelinversion.train import (
    SimpleTrainer,
    SimpleTrainConfig,
    VibTrainConfig,
    VibTrainer,
    BiDOTrainConfig,
    BiDOTrainer,
    TrapTrainConfig,
    TrapTrainer,
)
from modelinversion.utils import (
    Logger,
    LabelSmoothingCrossEntropyLoss,
    freeze_front_layers,
)
from modelinversion.datasets import FaceScrub64, FaceScrub224, CelebA64, CelebA224

facescrub_path = '/mnt/data/<usrname>/datasets/facescrub'
celeba_high_path = "/mnt/data/<usrname>/datasets/pre_celeba_high"
celeba_low_path = "/mnt/data/<usrname>/datasets/pre_celeba_low"
backbone_path = '/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/backbone/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

no_defense_model_path = "/mnt/data/<usrname>/mywork/lora_defense/checkpoints_v2/classifier/celeba64/celeba64_ir152_93.71.pth"

from torchvision.models import resnet18


def main(
    cuda_device: str,
    dataset_name: str,
    defense_name: str = None,
    defense_args: list[str] = None,
    tag_suffix: str = None,
    pretrain: bool = False,
):
    if isinstance(defense_args, str):
        defense_args = [defense_args]

    # num_classes = 530
    model_name = 'ir152'
    # tl_coef = 0.4
    if 'facescrub' in dataset_name.lower():
        dataset_path = facescrub_path
        num_classes = 530
    elif 'celeba' in dataset_name.lower():
        num_classes = 1000
        if '224' in dataset_name.lower():
            dataset_path = celeba_high_path
        else:
            dataset_path = celeba_low_path
    else:
        raise ValueError('invalid dataset name')

    save_tag = f'{dataset_name}_{model_name}'

    if defense_name is not None:

        save_tag = save_tag + f'_{defense_name}' + '_'.join(defense_args)

    if pretrain:
        save_tag = save_tag + '_pretrain'

    if tag_suffix is not None:
        save_tag = save_tag + '_' + tag_suffix

    save_name = f'{save_tag}.pth'
    experiment_dir = f'../result_classifier/train_{save_tag}'
    # backbone_path = '/data/<usrname>/Model-Inversion-Attack-ToolBox/checkpoints_v2/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'
    global backbone_path

    batch_size = 128
    epoch_num = 100

    device_ids_str = cuda_device
    pin_memory = False

    # prepare logger

    now_time = time.strftime(r'%Y%m%d_%H%M', time.localtime(time.time()))
    logger = Logger(experiment_dir, f'train_gan_{now_time}.log')

    # prepare devices

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # prepare target model

    # if pretrain:
    #     # model = IR152_64(num_classes, backbone_path=backbone_path)
    #     # model.load_state_dict(torch.load(no_defense_model_path)['state_dict'])
    #     # model.save_pretrained(no_defense_model_path.replace("_98.25", ""))
    #     # exit()
    #     model = auto_classifier_from_pretrained(no_defense_model_path)
    # else:
    model = IR152_64(num_classes, backbone_path=backbone_path)
    if pretrain:
        model.load_state_dict(
            torch.load(no_defense_model_path, map_location='cpu')['state_dict']
        )

    # exit()

    defense_name = defense_name.lower()
    if 'tl' in defense_name:
        tl_coef = float(defense_args[0])
        freeze_front_layers(model, ratio=tl_coef)
    elif 'vib' in defense_name:
        model = VibWrapper(model)
    elif 'bido' in defense_name:
        model = BiDOWrapper(model)
    elif 'rolss' in defense_name:
        model = SkipConnectionWrapper(
            model,
            residule_keep_ratio=float(defense_args[0]),
            erase_ratio_or_num=int(defense_args[1]),
        )

    # model = nn.DataParallel(model, device_ids=gpu_devices).to(device)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )
    torch.optim.Adam
    # prepare dataset

    dataset_name = dataset_name.lower()
    if 'facescrub' in dataset_name.lower():
        cls = FaceScrub64 if '64' in dataset_name else FaceScrub224
        train_dataset = cls(
            dataset_path,
            train=True,
            output_transform=Compose(
                [
                    ToTensor(),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
        test_dataset = cls(
            dataset_path,
            train=False,
            output_transform=Compose(
                [
                    ToTensor(),
                    # RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
    elif 'celeba' in dataset_name.lower():
        cls = CelebA64 if '64' in dataset_name else CelebA224
        train_dataset = cls(
            os.path.join(dataset_path, 'private_train'),
            output_transform=Compose(
                [
                    ToTensor(),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
        test_dataset = cls(
            os.path.join(dataset_path, 'private_test'),
            output_transform=Compose(
                [
                    ToTensor(),
                    # RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
    else:
        raise ValueError('invalid dataset name')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
    )

    # prepare train config

    loss_fn = 'ce'

    if 'vib' in defense_name:
        config = VibTrainConfig(
            experiment_dir=experiment_dir,
            save_name=save_name,
            device=device,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_schedular,
            loss_fn=loss_fn,
            beta=float(defense_args[0]),
        )
        trainer = VibTrainer(config)
    elif 'bido' in defense_name:
        config = BiDOTrainConfig(
            experiment_dir=experiment_dir,
            save_name=save_name,
            device=device,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_schedular,
            loss_fn=loss_fn,
            coef_hidden_input=float(defense_args[0]),
            coef_hidden_output=float(defense_args[1]),
        )
        trainer = BiDOTrainer(config)
    elif 'trap' in defense_name:
        raise NotImplementedError()
    else:
        config = SimpleTrainConfig(
            experiment_dir=experiment_dir,
            save_name=save_name,
            device=device,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_schedular,
            loss_fn=loss_fn,
        )

        trainer = SimpleTrainer(config)

    if not 'ls' in defense_name:
        trainer.train(epoch_num, train_loader, test_loader)
    else:
        ls_val = -float(defense_args[0])
        trainer.train(epoch_num // 2, train_loader, test_loader)
        for i in range(epoch_num // 4):
            trainer.loss_fn = LabelSmoothingCrossEntropyLoss(
                ls_val * (i / (epoch_num // 4))
            )
            trainer.train(1, train_loader, test_loader)
        trainer.train(epoch_num // 4, train_loader, test_loader)
    logger.close()


cuda = '0'
dataset_name = 'celeba64'

# main(cuda, dataset_name, 'vib', ['0.01'])

# main(cuda, dataset_name, 'bido', ['0.01', '0.1'])

# main(cuda, dataset_name, 'ls', ['0.3'])

# main(cuda, dataset_name, 'tl', ['0.5'])

# main(cuda, dataset_name, 'rolss', ['0.0', '2'])

# main(cuda, dataset_name, 'ls', ['0.05'])

main(cuda, dataset_name, 'ls', ['0.05'], pretrain=True)
main(cuda, dataset_name, 'rolss', ['0.0', '2'], pretrain=True)


main(cuda, dataset_name, 'bido', ['0.01', '0.1'], pretrain=True)
