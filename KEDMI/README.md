# Knowledge-Enriched-Distributional-Model-Inversion-Attacks

This is a PyTorch implementation of our paper at ICCV2021:

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

We propose a novel **'Inversion-Specific GAN'** that can better distill knowledge useful for performing attacks on private models from public data. Moreover,  we propose to *model a private data distribution* for each target class which refers to **'Distributional Recovery'**.

## Requirement
This code has been tested with Python 3.6, PyTorch 1.0 and cuda 10.0. 

## Getting Started
* Install required packages.
* Download relevant datasets including Celeba, MNIST, CIFAR10.
* Get target model prepared or run our code
    `python train_classifier.py` <br>
    Note that this code only provides three model architectures: VGG16, IR152, Facenet. And pretrained checkpoints for the three models can be downloaded at https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN?usp=sharing.

## Build a inversion-specific GAN
* Modify the configuration in 'celeba.json'.
* Modify the target model path in 'k+1_gan.py' to your customized path.
* Run
    `python k+1_gan.py`.
* Model checkpoints and generated image results are saved in folder ’improvedGAN‘.
* A general GAN can be obtained as a baseline by running
    `python binary_gan.py`.
* Pretrained binary GAN and inversion-specific GAN can be downloaded at https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing.


## Distributional Recovery
Run
    `python recovery.py`
    
* `--model` chooses the target model to attack.
* `--improved_flag` indicates if an inversion-specfic GAN is used. If False, then a general GAN will be applied.
* `--dist_flag` indicates if distributional recovery is performed. If False, then optimization is simply applied on a single sample instead of a distribution.
* By setting both `improved_flag` and `dist_flag` be False, we are simply using the method proposed in [[1]](#1).


## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
