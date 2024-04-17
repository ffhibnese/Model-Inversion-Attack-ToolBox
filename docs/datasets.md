
# Datasets

Here are the details for preprocessing datasets in 2 steps. We provide the preprocess tools for
+ celeba
+ facescrub
+ ffhq64
+ ffhq256
+ metfaces256
+ afhqdog256

Note that when using the `celeba64` and `facescrub64` datasets you can directly use the transform `Resize((64,64))` in torchvision on `celeba112` and `facescrub112` datasets respectively.


## Step 1: Download datasets

### Celeba

Download celeba dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The structure of the dataset is as follows:
```
<DOWNLOAD_PATH>
├── img_align_celeba
├── identity_CelebA.txt
├── list_attr_celeba.txt
├── list_bbox_celeba.txt
├── list_eval_partition.txt
├── list_landmarks_align_celeba.txt
└── list_landmarks_celeba.txt
```

For `celeba` with low resolution, you can directly use your download file above for step 3.

For `celeba` with high resolution (e.g. $224\times 224$), you need to follow [HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper) to increase the resolution of the cropped and aligned samples. Run the script of the cropper and replace all the images in `img_align_celeba`.
```sh
python align.py --crop_size_h 224 --crop_size_w 224 --order 3 --save_format png --face_factor 0.65 --n_worker 32
```

### FaceScrub

Use [this script](https://github.com/faceteam/facescrub) to download facescrub and some links are unavailable.

The structure of the dataset is as follows:
```
<DOWNLOAD_PATH>
├── actors
│   └── faces
└── actresses
    └── faces
```

### FFHQ

For `ffhq64`, download [thumbnails128x128](https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv).

For `ffhq256`, download [images1024x1024](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL).

### MetFaces

Download [here](https://drive.google.com/drive/folders/1iChdwdW7mZFUyivKtDwL8ehCNhYKQz6D).

### afhqdog

Follow [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) to download afhqdog dataset.
```sh
python dataset_tool.py --source=~/downloads/afhq/train/dog --dest=~/datasets/afhqdog.zip
```

## Step 2: Preprocess data

Fill the relative path for relative scripts in [examples/standard/datasets](../examples/standard/datasets) and run the scripts. Note that **FaceScrub dataset do not need to be preprocessed**. The parameters are as follows:
+ src_path: The path for the dataset you download.
+ dst_path: The path for the preprocessed dataset.
+ split_file_path: Only `celeba` need this parameter. We provide split files to split the dataset into train and test subset for `celeba`. Split files are available at [here](https://drive.google.com/drive/folders/13jGV8bsQnxZRMPSVOLzu3OVGWyQf5kpI). Note that you need to unzip the file.

The file structure of split files for `celeba` is as follows: 
```
split_files/
├── private_test.txt
├── private_train.txt
└── public.txt
```