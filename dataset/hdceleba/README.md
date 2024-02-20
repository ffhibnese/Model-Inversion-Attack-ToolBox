# hdceleba

`hdceleba` is a high-resolution version of `celeba`.

## Prepare dataset

[HD-CelebA Cropper](https://github.com/LynnHo/HD-CelebA-Cropper) can be used to increase the resolution of the cropped and aligned samples.

Follow the guidance of [HD-CelebA Cropper](https://github.com/LynnHo/HD-CelebA-Cropper) to prepare the dataset and run the align script:
```sh
python align.py --crop_size_h 224 --crop_size_w 224 --order 3 --save_format png --face_factor 0.65 --n_worker 32
```


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

Check the path of the prepared dataset `<HDCELEBA_DATASET_DIR>` and run this script to split the dataset into public and private parts.
```sh
python celeba_split.py <HDCELEBA_DATASET_DIR>
```