

# Parameters of attack configs

The parameters of attack configs can be devided into four main parts. 
+ [Folders](#folders)
+ [Target and eval models](#target-and-eval-models)
+ [Attack GAN models](#attack-gan-models)
+ [Misc](#misc)

## Folders

Relevant parameters:
+ `cache_dir`: The folder that store the intermediate results. Defaults to `./cache/<attack_method>`.
+ `result_dir`: The folder that store the final results. Defaults to `./result/<attack_method>`.
+ `ckpt_dir`: The folder that store the pre-trained models. Defaults to `./checkpoints`.
+ `dataset_dir`: The folder that store the datasets. Defaults to `./datasets`.

## Target and Eval Models

Checkpoints of some target and eval models can be installed from [here](https://drive.google.com/drive/folders/1uckndVVFB095w8MCTGLSqdq9wQ4WSvya?usp=drive_link). Place them to `<ckpt_dir>/<dataset-name>/`.

Relevant parameters:
+ `dataset_name`: The dataset that target/eval models trained on.
+ `target_name`: The name of target model (victim model).
+ `eval_name`: The name of evaluation model.

To use models we provided, legal values are as follows.

### Supported Pre-trained Target and Eval Model

The table below represents the pre-trained target and eval models we provided for each dataset. 

|             | celeba   | vggface2   | facescrub   |
|:----------------------:|:---------:|:-----------:|:------------:|
| vgg16                 | √        |            |             |
| ir152                 | √        |            |             |
| facenet64             | √        |            |             |
| facenet               | √        |            |             |
| resnet50_scratch_dag  |          | √          |             |
| inception_resnetv1    |          | √          |             |
| MobileNet             |          |            | √           |
| BackboneMobileFaceNet |          |            | √           |

### Supported Target/Eval Datasets For Each Attack Method

The table below represents the datasets supports for each attack method. Target and eval models supported for those datasets can be used.


|   | celeba   | vggface2   | facescrub   | imagenet   |
|:--------------:|:---------:|:-----------:|:------------:|:-----------:|
| C2FMI         |          |            | √           |            |
| DeepInversion |          |            |             | √          |
| GMI           | √        |            |             |            |
| KEDMI         | √        |            |             |            |
| PLGMI         | √        |            |             |            |
| Mirror        | √        | √          |             |            |
| BREPMI        | √        |            |             |            |

Note: You can use any ImageNet models supported by [torchvision](https://pytorch.org/vision/stable/models.html), which will be automatically downloaded when it is used.


## Attack GAN Models

The pre-trained GAN checkpoints should be placed in `<ckpt_dir>/<attack-method>/`.

At this stage, we do not provide codes for training attack GAN models. We will provided the training codes in MIA 2.0.

### C2FMI



### DeepInversion

DeepInversion recovers images by optimizing directly on the original image. **NO** GAN model.

### GMI

GMI training the gan without the target model. 

The pre-trained GAN we provided are as follows:
+ celeba
+ ffhq
+ facescrub

Relevant parameters:
+ `gan_dataset_name`: We provide `celeba`, `ffhq`  and `facescrub`.

### KEDMI

Relevant parameters:
+ `gan_target_name`: Target models used when traing GAN. We only provide `vgg16`.
+ `gan_dataset_name`: We provide `celeba`, `ffhq`  and `facescrub`.

### BREPMI

BREPMI use the same settings as KEDMI. It use the GAN in `<ckpt_dir>/KEDMI`, so it is not neccessary to create a folder for `BREPMI` in checkpoint folder.

### PLGMI

Relevant parameters:
+ `gan_target_name`: Target models used when traing GAN. We only provide `vgg16`.
+ `gan_dataset_name`: We provide `celeba`, `ffhq`  and `facescrub`.

### Mirror

Relevant parameters:
+ `genforce_name`: Models provided by [genforce](https://github.com/genforce/genforce). It will be automatically downloaded when used.



## Misc

Other neccessary parameters:
+ `target_labels`
+ `device`
+ `batch_size`