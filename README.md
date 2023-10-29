# Model_Inversion_Attack_Box

A comprehensive benchmark for model inversion attacks, which is easy to get started.

## Introduction

> TODO

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

> TODO

## Environments

> TODO

## Download Checkpoints

Download pre-trained models from [here](https://drive.google.com/drive/folders/1ko8zAK1j9lTSF8FMvacO8mCKHY9evG9L) and place them in `./checkpoints/`. The structure is shown in `./checkpoints_structure.txt`.

Genforces models will be automatic downloaded when running scripts.

## Run Example

Examples of attack algorithms is in `./dev_scripts/`. 

Example for running PLGMI attack:
```sh
python dev_scripts/plgmi.py
```

Results are generated in `./results`