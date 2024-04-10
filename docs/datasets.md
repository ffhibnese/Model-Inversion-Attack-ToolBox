
# datasets

Here are the details for preprocessing datasets in 3 steps. We provide the preprocess tools for
+ celeba64
+ celeba224
+ facescrub224
+ ffhq64
+ ffhq256
+ metfaces256
+ afhqdog256

## Step 1: Download split files

We provide split files to split the dataset into train and test subset for `celeba` and `facescrub`. Split files are available at [here](https://drive.google.com/drive/folders/13jGV8bsQnxZRMPSVOLzu3OVGWyQf5kpI). Note that you need to unzip the file.

The file structure of `celeba` is as follows: 
```
split_files/
├── private_test.txt
├── private_train.txt
└── public.txt
```

and the file structure of `facescrub` is
```
split_files/
├── private_test.txt
└── private_train.txt
```


## Step 2: Download datasets

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

For `celeba64`, you can directly use your download file above for step 3.

For `celeba224`, you need to follow [HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper) to increase the resolution of the cropped and aligned samples. Run the script of the cropper and replace all the images in `img_align_celeba`.
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

## Step 3: Preprocess data

Fill the relative path for relative scripts in [examples/standard/datasets](../examples/standard/datasets) and run the scripts. The parameters are as follows:
+ src_path: The path for the dataset you download.
+ dst_path: The path for the preprocessed dataset.
+ split_file_path: The path of split files in step 1. Only `celeba` and `facescrub` need this parameter.
