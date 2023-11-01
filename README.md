# Model_Inversion_Attack_Box
> TODO: correct the following icons with the correct software version we used.

![Python 3.9](https://img.shields.io/badge/python-3.9-DodgerBlue.svg?style=plastic)
![Pytorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

[Yixiang Qiu*](final-solution), 
[Hongyao Yu*](https://github.com/Chrisqcwx),
[Hao Fang*](https://github.com/ffhibnese),
[Wenbo Yu](https://github.com/cswbyu),
[Bin Chen#](https://github.com/BinChen2021)

> TODO: figure out a beautiful name for our work.
> 
Welcome to **MIA**! This repository is a comprehensive open-source Python benchmark for model inversion attacks, which is well-organized and easy to get started. It includes uniform implementations of advanced and representative model inversion methods, formulating a unified and reliable framework for a convenient and fair comparison between different model inversion methods.

> TODO: yixiang, fill in the blank with your email address.
> 
If you have any concerns about our toolbox, feel free to contact us at ___, fang-h23@mails.tsinghua.edu.cn.

Also, you are always welcome to contribute and make this repository better! 


## Introduction

> TODO: Introduce Model Inversion Attack briefly

## Features
- Easy to get started.
- Provide all the pre-trained model files.
- Always up to date.
- Well organized and encapsulated.
- A unified and fair comparison between attack methods.

## Model Inversion Attacks

|Method|Paper|Source Code|Key Characteristics|Addition Notes|
|:-:|:-:|:-:|:-:|:-:|
|[C2FMI](./src/modelinversion/attack/C2FMI/)|C2FMI: Corse-to-Fine Black-box Model Inversion Attack [TDSC'2023](https://ieeexplore.ieee.org/abstract/document/10148574)||||
|[DeepInversion](./src/modelinversion/attack/DeepInversion/)| Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion [CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html)||||
|[GMI](./src/modelinversion/attack/GMI/)| The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks [CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html)||||
|[KEDMI](./src/modelinversion/attack/KEDMI/)|Knowledge-Enriched Distributional Model Inversion Attacks [ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.html)||||
|[Mirror](./src/modelinversion/attack/Mirror/)|MIRROR: Model Inversion for Deep LearningNetwork with High Fidelity [NDSS'2022](https://www.ndss-symposium.org/ndss-paper/auto-draft-203/)||||
|[PLGMI](./src/modelinversion/attack/PLGMI/)|Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network [AAAI'2023](https://ojs.aaai.org/index.php/AAAI/article/view/25442)||||
|[SecretGen](./src/modelinversion/attack/SecretGen/)|SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination [ECCV'2022](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_9#Abs1)||||
|[VMI](./src/modelinversion/attack/VMI/)|Variational Model Inversion Attacks [NeurIPS'2021](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html)||||

## Model Inversion Defences

|Method|Paper|Source Code|Key Characteristics|Addition Notes|
|:-:|:-:|:-:|:-:|:-:|

> Coming soon...

## Environments

> TODO: export the conda env file and give instructions to install the environment, including git clone xxx and conda env create -n xxx -f xxx.

## Pre-trained Model Checkpoints
We prepare all the related pre-trained model files in the following link.   
Download pre-trained models from [here](https://drive.google.com/drive/folders/1ko8zAK1j9lTSF8FMvacO8mCKHY9evG9L) and place them in `./checkpoints/`. The detailed file path structure is shown in `./checkpoints_structure.txt`.

Genforces models will be automatically downloaded by running the provided scripts.

## Run Example
We provide detailed running scripts of attack algorithms in `/dev_scripts/`.

An example of running the PLGMI attack is as follows:

> python dev_scripts/plgmi.py

Results are generated in ./results


## Citation
> Coming soon...
