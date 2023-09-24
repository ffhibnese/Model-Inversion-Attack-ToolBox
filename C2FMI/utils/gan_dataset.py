from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    # 根据索引将数据库中的二进制数据流输出为transform后的PIL图像数据
    # 注意，一般Dataset类应返回 img, label；而GAN不需要label
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(6)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class DatasetCustomSize(MultiResolutionDataset):
    def __init__(self, img_num, path, transform, resolution=256):
        super(DatasetCustomSize, self).__init__(path, transform, resolution)
        self.img_num = img_num

    def __len__(self):
        return self.img_num
