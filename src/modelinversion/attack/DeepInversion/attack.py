from .config import DeepInversionConfig
from .code.imagenet_inversion import deepinversion_attack, DeepInversionArgs
import os
from ...foldermanager import FolderManager
import torchvision.models as tv_models
from ...models import get_model


def attack(config: DeepInversionConfig):
    # print(type(config))
    dataset_name = config.dataset_name
    assert dataset_name == 'imagenet'

    cache_dir = os.path.join(config.cache_dir, config.target_name)
    save_dir = os.path.join(config.result_dir, config.target_name)

    folder_manager = FolderManager(None, None, cache_dir, save_dir)
    target_model = get_model(
        config.target_name, config.dataset_name, device=config.device
    )
    eval_model = get_model(config.eval_name, config.dataset_name, device=config.device)

    args = DeepInversionArgs(
        adi_scale=config.adi_scale,
        device=config.device,
        bs=config.batch_size,
        lr=config.lr,
        target_name=config.target_name,
        eval_name=config.eval_name,
        # dataset_name = config.dataset_name,
        do_flip=config.do_flip,
        r_feature=config.r_feature,
        target_labels=config.target_labels,
    )

    # print(config)
    deepinversion_attack(args, target_model, eval_model, folder_manager)
