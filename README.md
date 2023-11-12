# :fire: Model Inversion Attack Box v1.0 :fire:

![Python 3.10](https://img.shields.io/badge/python-3.10-DodgerBlue.svg?style=plastic)
![Pytorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-DodgerBlue.svg?style=plastic)
![torchvision 0.15.2](https://img.shields.io/badge/torchvision-0.15.2-DodgerBlue.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-DodgerBlue.svg?style=plastic)

[Yixiang Qiu*](https://github.com/final-solution), 
[Hongyao Yu*](https://github.com/Chrisqcwx),
[Hao Fang*](https://github.com/ffhibnese),
[Wenbo Yu](https://github.com/cswbyu),
[Bin Chen#](https://github.com/BinChen2021),
[Shu-Tao Xia](https://www.sigs.tsinghua.edu.cn/xst/main.htm)

Welcome to **MIA**! This repository is a comprehensive open-source Python benchmark for model inversion attacks, which is well-organized and easy to get started. It includes uniform implementations of advanced and representative model inversion methods, formulating a unified and reliable framework for a convenient and fair comparison between different model inversion methods.


If you have any concerns about our toolbox, feel free to contact us at 200110409@stu.hit.edu.cn, 210110821@stu.hit.edu.cn, and fang-h23@mails.tsinghua.edu.cn.

Also, you are always welcome to contribute and make this repository better! 


### :construction: MIA v2.0 is coming soon
We are already in the second development stage where the following updates will be implemented soon.
- Representative defense algorithms
- Able to train your own generative model
- Better refactor code with `trainer`
- A package that can be installed with pip

## :rocket: Introduction

**Model inversion attack** is an emerging powerful private data theft attack, where a malicious attacker is able to reconstruct data with the same distribution as the training dataset of the target model. 

The reason why we developed this toolbox is that the research line of **MI** suffers from a lack of unified standards and reliable implementations of former studies. We hope our work can further help people in this area and promote the progress of their valuable research.

## :bulb: Features
- Easy to get started.
- Provide all the pre-trained model files.
- Always up to date.
- Well organized and encapsulated.
- A unified and fair comparison between attack methods.

## :memo: Model Inversion Attacks

|Method|Paper|Publication|Whitebox or Blackbox|Key Characteristics|
|:-:|:-:|:-:|:-:|:-:|
|[C2FMI](./src/modelinversion/attack/C2FMI/)|C2FMI: Corse-to-Fine Black-box Model Inversion Attack|[TDSC'2023](https://ieeexplore.ieee.org/abstract/document/10148574)|whitebox, blackbox|gradient-free, two-stage|
|[DeepInversion](./src/modelinversion/attack/DeepInversion/)|Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion|[CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html)|whitebox|student-teacher, data-free|
|[GMI](./src/modelinversion/attack/GMI/)|The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks|[CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html)|whitebox|the first GAN-based MIA, instance-level|
|[KEDMI](./src/modelinversion/attack/KEDMI/)|Knowledge-Enriched Distributional Model Inversion Attacks|[ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.html)|whitebox|the first MIA that recovers data distributions, pseudo-labels|
|[BREPMI](./src/modelinversion/attack/BREPMI/)|Label-Only Model Inversion Attacks via Boundary Repulsion|[CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kahla_Label-Only_Model_Inversion_Attacks_via_Boundary_Repulsion_CVPR_2022_paper.html)|blackbox|boundary repelling, label-only|
|[Mirror](./src/modelinversion/attack/Mirror/)|MIRROR: Model Inversion for Deep Learning Network with High Fidelity|[NDSS'2022](https://www.ndss-symposium.org/ndss-paper/auto-draft-203/)|whitebox, blackbox|both gradient-free and gradient-based, genetic algorithm|
|[PLGMI](./src/modelinversion/attack/PLGMI/)|Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network|[AAAI'2023](https://ojs.aaai.org/index.php/AAAI/article/view/25442)|whitebox|pseudo-labels, data augmentation, special loss function|
|[SecretGen](./src/modelinversion/attack/SecretGen/)|SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination|[ECCV'2022](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_9#Abs1)|whitebox, blackbox|instance-level, data augmentation|
|[VMI](./src/modelinversion/attack/VMI/)|Variational Model Inversion Attacks|[NeurIPS'2021](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html)|whitebox|variational inference, special loss function|

## :memo: Model Inversion Defenses

|Method|Paper|Publication|Key Characteristics|
|:-:|:-:|:-:|:-:|
|[ViB / MID](./src/modelinversion/defense/Vib/)|Improving Robustness to Model Inversion Attacks via Mutual Information Regularization|[AAAI'2021](https://arxiv.org/abs/2009.05241)| variational method, mutual information, special loss function|
|[BiDO](./src/modelinversion/defense/BiDO/)|Bilateral Dependency Optimization: Defending Against Model-inversion Attacks|[KDD'2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539376)|special loss function|

## :wrench: Environments
**MIA** can be built up with the following steps:
1. Clone this repository and create the virtual environment by Anaconda.
```sh
git clone https://github.com/ffhibnese/Model_Inversion_Attack_Box.git
cd ./Model_Inversion_Attack_Box
conda create -n MIA python=3.10
```
2. Install the related dependencies:
```sh
pip install -r requirements.txt
```

## :page_facing_up: Datasets and Model Checkpoints
- For datasets, you can download them according to the file placed in `./dataset/<DATASET_NAME>/README.md`.
- For pre-trained models, we prepare all the related model weights files in the following link.   
Download pre-trained models [here](https://drive.google.com/drive/folders/1ko8zAK1j9lTSF8FMvacO8mCKHY9evG9L) and place them in `./checkpoints/`. The detailed file path structure is shown in `./checkpoints_structure.txt`.

Genforces models will be automatically downloaded by running the provided scripts.

## :racehorse: Run Examples
We provide detailed running scripts of attack algorithms in `./dev_scripts/`.
You can run any attack algorithm simply by the following instruction and experimental results will be produced in `./results/<ATTACK_METHOD>/` by default:
```sh
python dev_scripts/<ATTACK_METHOD>.py
```


## 📔 Citation
> Coming soon...

## :sparkles: Acknowledgement
We express great gratitude for all the researchers' contributions to the **Model Inversion** community. 

In particular, we thank the authors of [PLGMI](https://github.com/LetheSec/PLG-MI-Attack) for their high-quality codes for datasets, metrics, and three attack methods. It's their great devotion that helps us make **MIA** better!  
