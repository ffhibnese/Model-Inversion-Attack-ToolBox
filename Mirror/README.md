# Official Code for MIRROR (NDSS 2022)

This is the PyTorch implementation for NDSS 2022 paper "MIRROR: Model Inversion for Deep Learning Network with High Fidelity". For more inversion results, please refer to [mirror github page](https://model-inversion.github.io/mirror/).

Note: we only uploaded the scripts and cache files for resnet50 and vgg16 model. Others will be updated soon.

## Environment

```
conda env create -f environment.yml
conda activate mirror
```

## Usage

### 1. Download cache files

```
python my_download_cache.py
```

### 2. White-box invert ResNet50

Conduct the inversion.

```
zsh scripts/run_vggface2_resnet50_whitebox.sh

```

Test the inversion.

```
zsh scripts/run_vggface2_resnet50_whitebox_test.sh
```

### 3. Black-box invert ResNet50

Conduct the inversion.

```
zsh scripts/run_vggface2_resnet50_blackbox.sh

```

Test the inversion.

```
zsh scripts/run_vggface2_resnet50_blackbox_test.sh
```

## Build cache files

We need to first generate styelgan's images if we haven't done it.

```
python my_sample_z_w_space.py
```

Use inception_resnetv1_vggface2 as an example. We first generate the outputs of the network on the stylegan's samples. Then we merge them into one file.

```
python my_generate_blackbox_attack_dataset.py --arch_name inception_resnetv1_vggface2 stylegan
python my_merge_all_tensors.py blackbox_attack_data/stylegan/inception_resnetv1_vggface2/no_dropout --remove
```

## Acknowledgement

The StyleGAN models are based on [genforce/genforce](https://github.com/genforce/genforce).

VGG16/VGG16BN/Resnet50 models are from [their official websites](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html).

InceptionResnetV1 is from [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch).

SphereFace is from [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch).


## BibTex
Please cite our work as follows for any purpose of usage.

```
@inproceedings{An.Mirror.NDSS.2022,
    title={MIRROR: Model Inversion for Deep Learning Network with High Fidelity},
    author={An, Shengwei and Tao, Guanhong and Xu, Qiuling and Liu, Yingqi and Shen, Guangyu and Yao, Yuan and Xu, Jingwei and Zhang, Xiangyu},
    booktitle={Proceedings of the Network and Distributed Systems Security Symposium (NDSS 2022)},
    year={2022}
}
```
