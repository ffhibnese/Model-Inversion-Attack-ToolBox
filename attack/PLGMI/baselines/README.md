# Baselines

The code is adapted from: https://github.com/SCccc21/Knowledge-Enriched-DMI

The checkpoints can be found at: https://drive.google.com/drive/folders/1qDvl7i6_U7xoaduUbeEzTSuYXWpxvvXt?usp=sharing

## [GMI](https://arxiv.org/abs/1911.07135)

To train the GAN of GMI, please run the `GMI.py` as follows:

```bash
python GMI.py --dataset_name=ffhq --save_dir=GMI_Baseline
```

To reconstruct the private images of using the trained generator, pealse run the `recovery.py` as follows:

```bash
python recovery.py \
--model VGG16 \
--save_dir=GMI_Inversion \
--path_G=./GMI_Baseline/ffhq/ffhq_GMI_G.tar \
--path_D=./GMI_Baseline/ffhq/ffhq_GMI_D.tar
```

## [KED-MI](https://arxiv.org/abs/2010.04092)

To train the GAN of KED-MI, please run the `KED_MI.py` as follows:

```bash
python KED_MI.py --model_name_T=VGG16 --dataset_name=ffhq --save_dir=KED_MI_Baseline
```

To reconstruct the private images of using the trained generator, pealse run the `recovery.py` as follows:

```bash
python recovery.py \
--model VGG16 \
--improved_flag \
--dist_flag \
--save_dir=KED_MI_Inversion \
--path_G=./KED_MI_Baseline/ffhq/VGG16/ffhq_VGG16_KED_MI_G.tar \
--path_D=./KED_MI_Baseline/ffhq/VGG16/ffhq_VGG16_KED_MI_D.tar
```


