

# Parameters of attack configs

The parameters of attack configs can be devided into three main parts. 
+ [Folders](#folders)
+ [Target and eval models](#target-and-eval-models)
+ [Specific parameters of attack methods](#attack-methods)

## Folders



## Target and Eval Models

Checkpoints of some target and eval models can be installed from [here](https://drive.google.com/drive/folders/1uckndVVFB095w8MCTGLSqdq9wQ4WSvya?usp=drive_link). Place them to `<ckpt_dir>/<dataset-name>/`.

Some parameters in attack configs:
+ `dataset_name`: The dataset that target/eval models trained on.
+ `target_name`: The name of target model (victim model).
+ `eval_name`: The name of evaluation model.

Legal values are as follows.

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


## Attack Methods

At this stage, we do not provide codes for training attack models. We will provided the training codes in MIA 2.0.

# TODO

## C2FMI

dataset_name - target_name/eval_name:

+ FaceScrub
    + target / eval models for FaceScrub

## DeepInversion

target_name/eval_name:
+ models provided by [torchvision](https://pytorch.org/vision/0.15/models.html)

## GMI

dataset_name - target_name/eval_name:

+ celeba
    + target / eval models for celeba

gan_dataset_name - gan_target_name:

+ celeba
    + vgg16
+ ffhq
    + vgg16
+ facescrub
    + vgg16

## KEDMI

dataset_name - target_name/eval_name:

+ celeba
    + target / eval models for celeba

gan_dataset_name - gan_target_name:

+ celeba
    + vgg16
+ ffhq
    + vgg16
+ facescrub
    + vgg16

## PLGMI

dataset_name - target_name/eval_name:

+ celeba
    + target / eval models for celeba

gan_dataset_name - gan_target_name:

+ celeba
    + vgg16
+ ffhq
    + vgg16
+ facescrub
    + vgg16

## Mirror

dataset_name - target_name/eval_name:

+ celeba
    + target / eval models for celeba
+ vggface2:
    + target / eval models for vggface2

genforce_name:

+ models provided by [genforce](https://github.com/genforce/genforce)



