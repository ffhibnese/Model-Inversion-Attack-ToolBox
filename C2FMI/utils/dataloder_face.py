from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class FacenetDataset(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_classes):
        super(FacenetDataset, self).__init__()

        self.img_height   = input_shape[0]
        self.img_width    = input_shape[1]
        self.channel      = input_shape[2]
        self.dataset_path = dataset_path   # ['label1;abs_path1\n','labeln;abs_path2\n',...]
        self.num_train    = num_train
        self.num_classes  = num_classes

        self.paths  = []
        self.labels = []
        self.load_dataset()

    def load_dataset(self):
        for label_path in self.dataset_path:
            d_split = label_path.split(';')
            self.labels.append(int(d_split[0]))
            self.paths.append(d_split[1].split()[0])  # .split()默认对空字符（空格、换行\n、制表\t）进行split
        self.labels = np.array(self.labels)
        self.paths  = np.array(self.paths, dtype=np.object)

    def get_processed_img(self, image, jitter=.1, hue=.05, sat=1.3, val=1.3, flip_signal=True):
        # 对图像进行一系列处理并返回
        image = image.convert("RGB")

        h, w = self.img_height, self.img_width
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.9,1.1)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        flip = rand()<.5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand()<.5
        if rotate:
            angle=np.random.randint(-5,5)
            a,b=w/2,h/2
            M=cv2.getRotationMatrix2D((a,b),angle,1)
            image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[128,128,128])

        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        if self.channel==1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
        # cv2.waitKey(0)
        return image_data

    def __len__(self):
        # DataLoader对index从0->len进行sample
        return self.num_train

    def __getitem__(self, index):
        # 三张图片 anchor、positive、negative为一组
        images = np.zeros((3, self.channel, self.img_height, self.img_width))
        labels = np.zeros((3))

        # 选择anchor和positive
        rand_label    = np.random.randint(0, self.num_classes)
        selected_path = self.paths[self.labels == rand_label]
        while len(selected_path) < 2:
            rand_label    = np.random.randint(0, self.num_classes)
            selected_path = self.paths[self.labels == rand_label]

        # 选择 anchor
        img_indexes = np.random.choice(range(0, len(selected_path)), 2)
        image = Image.open(selected_path[img_indexes[0]])
        image = self.get_processed_img(image)
        image = np.transpose(np.asarray(image).astype(np.float64), [2,0,1]) / 255  # H*W*C -> C*H*W
        if self.channel == 1:
            images[0,0,:,:] = image
        else:
            images[0,:,:,:] = image
        labels[0] = rand_label

        # 选择 positive
        image = Image.open(selected_path[img_indexes[1]])
        image = self.get_processed_img(image)
        image = np.transpose(np.asarray(image).astype(np.float64), [2,0,1]) / 255  # H*W*C -> C*H*W
        if self.channel == 1:
            images[1,0,:,:] = image
        else:
            images[1,:,:,:] = image
        labels[1] = rand_label

        # 选择 negative
        diff_labels = list(range(self.num_classes))
        diff_labels.pop(rand_label)
        tmp_index = np.random.choice(range(self.num_classes-1), 1)
        diff_rand_label = diff_labels[tmp_index[0]]
        diff_selected_path = self.paths[self.labels == diff_rand_label]
        while len(diff_selected_path) < 1:
            tmp_index = np.random.choice(range(self.num_classes-1), 1)
            diff_rand_label = diff_labels[tmp_index[0]]
            diff_selected_path = self.paths[self.labels == diff_rand_label]

        img_index = np.random.choice(range(0, len(diff_selected_path)), 1)
        image = Image.open(diff_selected_path[img_index[0]])
        image = self.get_processed_img(image)
        image = np.transpose(np.asarray(image).astype(np.float64), [2,0,1]) / 255  # H*W*C -> C*H*W
        if self.channel == 1:
            images[2,0,:,:] = image
        else:
            images[2,:,:,:] = image
        labels[2] = diff_rand_label

        return images, labels

def collate_fc():
    def dataset_collate(batch):
        # batch: DataLoader采样器采样出来的batch_size个(images, labels)
        # return: DataLoder的一个输出batch: 3batch_size*C*H*W

        images = []
        labels = []
        for img, label in batch:
            images.append(img)      # img为3*C*H*W, images为batch_size*3*C*H*W
            labels.append(label)    # labels为batch_size*3
        images = np.array(images)
        labels = np.array(labels)

        images1 = images[:,0,:,:,:]  # batch_size*C*H*W
        images2 = images[:,1,:,:,:]  # batch_size*C*H*W
        images3 = images[:,2,:,:,:]  # batch_size*C*H*W
        images  = np.concatenate([images1, images2, images3], 0)

        labels1 = labels[:,0]
        labels2 = labels[:,1]
        labels3 = labels[:,2]
        labels  = np.concatenate([labels1, labels2, labels3], 0)

        # 返回 3batch_size*C*H*W 的images，其中[0:batch_size]为anchors，[batch_size:2batch_size]为positives，[2batch_size:]为negatives
        return images, labels
    return dataset_collate
