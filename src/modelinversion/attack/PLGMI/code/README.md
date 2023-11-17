# Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network

Code for the paper **<a href="https://arxiv.org/abs/2302.09814">Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network</a> (AAAI
2023)**. 


![framework](imgs/framework.jpg)

# Requirement

Install the environment as follows:

```bash
# create conda environment
conda create -n PLG_MI python=3.9
conda activate PLG_MI
# install pytorch 
conda install pytorch==1.10.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# install other dependencies
pip install -r requirements.txt
```

# Preparation

## Dataset

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  , [FFHQ](https://drive.google.com/open?id=1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv)
  and [FaceScrub](http://vintage.winklerbros.net/facescrub.html) are used for expriments (we
  use [this script](https://github.com/faceteam/facescrub) to download FaceScrub and some links are unavailable.)

- We follow the [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) to divide the CelebA into the private data
  and the public data. The private data of CelebA can be found
  at: https://drive.google.com/drive/folders/1uxSsbNwCKZcy3MQ4mA9rpwiJRhtpTas6?usp=sharing
- You should put them as follows:
    ```
    datasets
    ├── celeba
    │   └── img_align_celeba
    ├── facescrub
    │   └── faceScrub
    ├── ffhq
    │   └── thumbnails128x128
    └── celeba_private_domain
    ````

## Models

- You can train target models following [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) or direcly download
  the provided checkpoints at: https://drive.google.com/drive/folders/1Cf2O2MVvveXrBcdBEWDi-cMGzk0y_AsT?usp=sharing and
  put them in folder `./checkpoints`.

- To calculate the KNN_dist, we get the features of private data on the evaluation model in advance. You can download
  at: https://drive.google.com/drive/folders/1Aj9glrxLoVlfrehCX2L9weFBx5PK6z-x?usp=sharing and put them in
  folder `./celeba_private_feats`.

# Top-n Selection Strategy

To get the pseudo-labeled public data using top-n selection strategy, pealse run the `top_n_selection.py` as follows:

```bash
python top_n_selection.py --model=VGG16 --data_name=ffhq --top_n=30 --save_root=reclassified_public_data
```

# Pseudo Label-Guided cGAN

To train the conditional GAN in stage-1, please run the `train_cgan.py` as follows:

```bash
python train_cgan.py \
--data_name=ffhq \
--target_model=VGG16 \
--calc_FID \
--inv_loss_type=margin \
--max_iteration=30000 \
--alpha=0.2 \
--private_data_root=./datasets/celeba_private_domain \
--data_root=./reclassified_public_data/ffhq/VGG16_top30 \
--results_root=PLG_MI_Results
```

The checkpoints can be found at: https://drive.google.com/drive/folders/1qDvl7i6_U7xoaduUbeEzTSuYXWpxvvXt?usp=sharing

# Image Reconstruction

To reconstruct the private images of specified class using the trained generator, pealse run the `reconstruct.py` as
follows:

```bash
python reconstruct.py \
--model=VGG16 \
--inv_loss_type=margin \
--lr=0.1 \
--iter_times=600 \
--path_G=./PLG_MI_Results/ffhq/VGG16/gen_latest.pth.tar \
--save_dir=PLG_MI_Inversion
```

# Examples of reconstructed face images

![examples](imgs/examples.jpg)
