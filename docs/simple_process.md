# Simple Process

Here we take GMI as an example to introduce the whole process.

## Data Preparation

Follow [dataset.md](./datasets.md) to prepare the CelebA dataset.

The structure of the dataset should be like this:
```
<celeba_path>
├── public
├── private_train
└── private_test
```

## Classifier Training

Here we train IR152 as the target model and FaceNet112 as the eval model.

### IR152

To train the IR152 model with [classifier_training/celeba64.py](../examples/standard/classifier_training/celeba64.py) as an example, you can fill the paths in the script:
```python
save_name = f'<name>.pth'
train_dataset_path = '<celeba_path>/private_train'
test_dataset_path = '<celeba_path>/private_test'
experiment_dir = '<ir152_train_result_dir>'
backbone_path = '<ir152_backbone_path>'
```

The `ir152_backbone_path` is the path to the pre-trained IR152 backbone model. You can download it from [Google Drive](https://drive.google.com/file/d/1qz6Z6X7Q1j7Q6j0VY9Zj1X0Zj1X0Zj1X/view?usp=sharing)/checkpoints_v2.0/classifier/backbones/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth.

The model will be saved in `<ir152_train_result_dir>/<name>.pth`, denoted as `<target_model_ckpt_path>` in the following text.

### FaceNet112

To train the FaceNet112 model with [classifier_training/celeba112.py](../examples/standard/classifier_training/celeba112.py) as an example, you can fill the paths in the script:
```python
save_name = f'<name>.pth'
train_dataset_path = '<celeba_path>/private_train'
test_dataset_path = '<celeba_path>/private_test'
experiment_dir = '<facenet112_train_result_dir>'
backbone_path = '<facenet112_backbone_path>'
```

The `facenet112_backbone_path` is the path to the pre-trained IR152 backbone model. You can download it from [Google Drive](https://drive.google.com/file/d/1qz6Z6X7Q1j7Q6j0VY9Zj1X0Zj1X0Zj1X/view?usp=sharing)/checkpoints_v2.0/classifier/backbones/backbone_ir50_ms1m_epoch120.pth.


The model will be saved in `<facenet112_train_result_dir>/<name>.pth`, denoted as `<eval_model_ckpt_path>` in the following text.

## GMI GAN training

To train the GMI GAN with [gan_training/gmi.py](../examples/standard/gan_training/gmi.py) as an example, you can fill the paths in the script:
```python
dataset_path = '<celeba_path>/public'
experiment_dir = '<gmi_gan_dir>'
```

The generator and discriminator will be saved in `<gmi_gan_dir>/G.pth` and `<gmi_gan_dir>/D.pth`, denoted as `<generator_ckpt_path>` and `<discriminator_ckpt_path>` in the following text.


## GMI Attack

The attack script is [attacks/gmi.py](../examples/standard/attacks/gmi.py). You can fill the paths in the script:
```python
experiment_dir = '<gmi_attack_result_dir>'
device_ids_available = '0'
num_classes = 1000
generator_ckpt_path = '<generator_ckpt_path>'
discriminator_ckpt_path = '<discriminator_ckpt_path>'
target_model_ckpt_path = '<target_model_ckpt_path>'
eval_model_ckpt_path = '<eval_model_ckpt_path>'
eval_dataset_path = '<celeba_path>/private_train'
```

The attack result will be saved in `<gmi_attack_result_dir>`.

Evaluation results are shown in `<gmi_attack_result_dir>/optimized/evaluation.csv` 