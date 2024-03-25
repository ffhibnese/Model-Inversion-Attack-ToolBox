# Celeba

## Download dataset

Download celeba dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Download split files

Open Terminal here and create a folder:
```sh
mkdir split_files
```

Download split files from [here](https://drive.google.com/drive/folders/1-40XCE3fMvHOxWNV8bjfmrAt856HGTAO?usp=drive_link) and place them into `./split_files`. The file structure is as follows: 

```
split_files/
├── private_test.txt
├── private_train.txt
└── public.txt
celeba_split.py
README.md
```

## Split files

Find folder `image_align_data` from the celeba dataset you downloaded and run:
```sh
python celeba_split.py <YOUR_DATASET_DIR>/img_align_celeba
```