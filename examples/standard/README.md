
# examples-standard

Here are some standard examples about how to use this toolbox.

## adapter_training

1. run the attack method of C2FMI:
```bash
python ./adapter_training/c2f.py
```

2. run the attack method of C2FMI on high-resolution images:
```bash
python ./adapter_training/c2f_high.py
```

## attacks

1. config the options in each attack script as instructed, such as in `brepmi.py`:
```python
experiment_dir = '<fill it>'
generator_ckpt_path = '<fill it>'
target_model_ckpt_path = '<fill it>'
eval_model_ckpt_path = '<fill it>'
eval_dataset_path = '<fill it>'
```

2. run the attack method of BREPMI, GMI, LOMMA, MIRROR, PLGMI, PPA, RLBMI, VIM:
```bash
python ./attacks/<ATTACK_METHOD_NAME>.py
```

## classifier_training

1. config the options in each training script as instructed, such as in `celeba64.py`:
```python
train_dataset_path = '<fill it>'
test_dataset_path = '<fill it>'
experiment_dir = '<fill it>'
backbone_path = '<fill it, or set as None>'
```

2. run the training scripts of classifiers under various resolutions:
```bash
python ./classifier_training/<TRAIN_METHOD_NAME>.py
```

## dataset_preprocess

1. config the options in each preprocess script as instructed, such as in `afhqdogs256.py`:
```python
src_path = '<fill it>'
dst_path = '<fill it>'
```

2. run the preprocess scripts of datasets under various resolutions:
```bash
python ./dataset_preprocess/<PREPROCESS_METHOD_NAME>.py
```

## gan_training

1. config the options in each gan training script as instructed, such as in `gmi.py`:
```python
dataset_path = '<fill it>'
experiment_dir = '<fill it>'
```

2. run the preprocess scripts of datasets under various resolutions:
```bash
python ./gan_training/<GAN_TRAIN_METHOD_NAME>.py
```
