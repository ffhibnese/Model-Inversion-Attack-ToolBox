import matplotlib.pylab as plt
import pandas
import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

CHESTXRAY_ROOT = '/scratch/hdd001/home/wangkuan/data/chestxray'
CLASS_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
]


def load_ims_by_fnames(fnames, tr):
    ims = []
    for fname in tqdm(fnames, desc='load im'):
        fpath = os.path.join(CHESTXRAY_ROOT, 'images', fname)
        im = plt.imread(fpath)
        if len(im.shape) == 3 and im.shape[-1] == 4:  # (D, D, 4)
            im = im[:, :, :3].mean(-1)
        try:
            im = tr(im)
        except:
            import ipdb

            ipdb.set_trace()
        ims.append(im)
    ims = torch.stack(ims)
    return ims


def create_data_cache(image_size):
    size_folder = f"{image_size}x{image_size}"
    os.makedirs(os.path.join(CHESTXRAY_ROOT, size_folder), exist_ok=True)

    tr = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size)),
            transforms.ToTensor(),
        ]
    )

    dfpath = os.path.join(CHESTXRAY_ROOT, 'Data_Entry_2017_v2020.csv')
    df = pandas.read_csv(dfpath)

    test_fnames = pandas.read_csv(
        os.path.join(CHESTXRAY_ROOT, 'test_list.txt'), names=['fnames']
    )['fnames'].values
    train_fnames = pandas.read_csv(
        os.path.join(CHESTXRAY_ROOT, 'train_val_list.txt'), names=['fnames']
    )['fnames'].values

    # Collect Train/Test fnames by class
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for c in tqdm(range(8), desc='loading images by class'):
        fnames_c = df[df['Finding Labels'] == CLASS_LABELS[c]]['Image Index'].values
        fnames_c_train = list(set(fnames_c).intersection(set(train_fnames)))
        fnames_c_test = list(set(fnames_c).intersection(set(test_fnames)))
        train_x.append(load_ims_by_fnames(fnames_c_train, tr))
        test_x.append(load_ims_by_fnames(fnames_c_test, tr))
        train_y.append(c * torch.ones(len(fnames_c_train)))
        test_y.append(c * torch.ones(len(fnames_c_test)))
    train_x = torch.cat(train_x)
    test_x = torch.cat(test_x)
    train_y = torch.cat(train_y)
    test_y = torch.cat(test_y)

    np.save(
        open(os.path.join(CHESTXRAY_ROOT, size_folder, 'train_x.npy'), 'wb'),
        (train_x.numpy() * 255).astype('uint8'),
    )
    np.save(
        open(os.path.join(CHESTXRAY_ROOT, size_folder, 'train_y.npy'), 'wb'),
        (train_y.numpy()).astype('uint8'),
    )
    np.save(
        open(os.path.join(CHESTXRAY_ROOT, size_folder, 'test_x.npy'), 'wb'),
        (test_x.numpy() * 255).astype('uint8'),
    )
    np.save(
        open(os.path.join(CHESTXRAY_ROOT, size_folder, 'test_y.npy'), 'wb'),
        (test_y.numpy()).astype('uint8'),
    )


def create_aux_data_cache(image_size):
    size_folder = f"{image_size}x{image_size}"
    os.makedirs(os.path.join(CHESTXRAY_ROOT, size_folder), exist_ok=True)

    tr = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size)),
            transforms.ToTensor(),
        ]
    )

    dfpath = os.path.join(CHESTXRAY_ROOT, 'Data_Entry_2017_v2020.csv')
    df = pandas.read_csv(dfpath)

    # Collect Train/Test fnames by class
    mask = df['Finding Labels'] != CLASS_LABELS[0]
    for c in range(8):
        mask = mask * df['Finding Labels'] != CLASS_LABELS[c]
    fnames = df[mask]['Image Index'].values
    np.random.seed(0)
    np.random.shuffle(fnames)
    x = load_ims_by_fnames(fnames[:50000], tr)

    np.save(
        open(os.path.join(CHESTXRAY_ROOT, size_folder, 'aux_x.npy'), 'wb'),
        (x.numpy() * 255).astype('uint8'),
    )


def load_data_cache(image_size):
    size_folder = f"{image_size}x{image_size}"
    train_x = (
        np.load(os.path.join(CHESTXRAY_ROOT, size_folder, 'train_x.npy')).astype(
            'float32'
        )
        / 255
    )
    train_y = np.load(os.path.join(CHESTXRAY_ROOT, size_folder, 'train_y.npy'))
    test_x = (
        np.load(os.path.join(CHESTXRAY_ROOT, size_folder, 'test_x.npy')).astype(
            'float32'
        )
        / 255
    )
    test_y = np.load(os.path.join(CHESTXRAY_ROOT, size_folder, 'test_y.npy'))

    # To Torch Tensor
    train_x = torch.from_numpy(train_x) * 2 - 1
    train_y = torch.from_numpy(train_y).long()
    test_x = torch.from_numpy(test_x) * 2 - 1
    test_y = torch.from_numpy(test_y).long()

    return train_x, train_y, test_x, test_y


def visualize_subset_by_labels():
    image_size = 128
    tr = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size)),
            transforms.ToTensor(),
        ]
    )

    labels = ['No Finding'] + CLASS_LABELS
    os.makedirs(os.path.join(CHESTXRAY_ROOT, 'viz'), exist_ok=True)

    dfpath = os.path.join(CHESTXRAY_ROOT, 'Data_Entry_2017_v2020.csv')
    df = pandas.read_csv(dfpath)

    for label in labels:
        # os.makedirs(os.path.join(CHESTXRAY_ROOT, 'viz', label), exist_ok=True)
        fnames_c = df[df['Finding Labels'] == label]['Image Index'].values
        fnames_c = fnames_c[:100]

        ims = load_ims_by_fnames(fnames_c, tr)

        fpath = os.path.join(CHESTXRAY_ROOT, 'viz', f'{label}.jpeg')
        vutils.save_image(ims, fpath, nrow=10)


if __name__ == '__main__':
    visualize_subset_by_labels()
    # create_data_cache(256)
    # create_aux_data_cache(256)

    # # Visualize Data
    # ims = []
    # for c in range(8):
    #     fnames = df[df['Finding Labels'] ==
    #                 CLASS_LABELS[c]]['Image Index'].values[:10]
    #     for fname in fnames:
    #         fpath = os.path.join(CHESTXRAY_ROOT, 'images', fname)
    #         im = plt.imread(fpath)
    #         im = tr(im)
    #         ims.append(im)
    # ims = torch.stack(ims)
    # vutils.save_image(ims, 'chests.jpeg', nrow=10)

    """Frequency of labels (top N):
    No Finding                                              60361
    Infiltration                                             9547
    Atelectasis                                              4215
    Effusion                                                 3955
    Nodule                                                   2705
    Pneumothorax                                             2194
    Mass                                                     2139
    Effusion|Infiltration                                    1603
    Atelectasis|Infiltration                                 1350
    Consolidation                                            1310
    Atelectasis|Effusion                                     1165
    Pleural_Thickening                                       1126
    Cardiomegaly                                             1093

    """
