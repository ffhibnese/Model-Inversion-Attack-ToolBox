import os
import yaml
from typing import Optional
from collections import OrderedDict
import torch
import pandas as pd


def safe_save(obj, save_dir: str, save_name: Optional[str] = None):
    """Save the obj by using torch.save function.

    Args:
        obj (_type_): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the objective to save. Defaults to None.
    """

    if save_name is None:
        save_dir, save_name = os.path.split(save_dir)
    if save_dir.strip() != '':
        os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, os.path.join(save_dir, save_name))


def safe_save_csv(df: pd.DataFrame, save_dir: str, save_name: Optional[str] = None):
    """Save the data in csv format.

    Args:
        df (pd.DataFrame): The data to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    if save_name is None:
        save_dir, save_name = os.path.split(save_dir)
    if save_dir.strip() != '':
        os.makedirs(save_dir, exist_ok=True)
    # torch.save(obj, os.path.join(save_dir, save_name))
    df.to_csv(os.path.join(save_dir, save_name), index=None)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def walk_imgs(path):
    """Traverse all images in the specified path.

    Args:
        path (_type_): The specified path.

    Returns:
        List: The list that collects the paths for all the images.
    """

    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(IMG_EXTENSIONS):
                img_paths.append(os.path.join(root, file))
    return img_paths


yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        'tag:yaml.org,2002:map', data.items()
    ),
)
yaml.add_representer(
    tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data)
)


def obj_to_yaml(obj) -> str:
    return yaml.dump(obj)


def print_as_yaml(obj, stdout=True, file=None, mode='w'):
    """Print the obj in the yaml format and Save the obj if the file path is specified.

    Args:
        obj (_type_): The objective to save.
        stdout (bool, optional): Whether to print in the stdout. Defaults to True.
        file (_type_, optional): The file path for the obj to save. Defaults to None.
        mode (str, optional): An optional string that specifies the mode in which the file is opened. Defaults to 'w'.
    """
    
    s = yaml.dump(obj)

    if stdout:
        print(s)
    if file:
        with open(file, mode) as f:
            f.write(s)


def print_split_line(content=None, length=60):
    """Print the content and surround it with '-' character for alignment.

    Args:
        content (_type_, optional): The content to print. Defaults to None.
        length (int, optional): The total length of content and '-' characters. Defaults to 60.
    """
    
    if content is None:
        print('-' * length)
        return
    if len(content) > length - 4:
        length = len(content) + 4

    total_num = length - len(content) - 2
    left_num = total_num // 2
    right_num = total_num - left_num
    print('-' * left_num, end=' ')
    print(content, end=' ')
    print('-' * right_num)
