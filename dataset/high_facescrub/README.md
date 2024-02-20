# high_facescrub

High quality version of facescrub

## Prepare dataset

Run the following script to download the dataset
```sh
mkdir -p raw_dataset/actors
mkdir -p raw_dataset/actresses
python python3_download_facescrub.py facescrub_actors.txt raw_dataset/actors/ --crop_face
python python3_download_facescrub.py facescrub_actresses.txt raw_dataset/actresses/ --crop_face
```