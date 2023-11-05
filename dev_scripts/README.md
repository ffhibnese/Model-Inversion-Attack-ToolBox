
# Support Models

Here are pretrained models for each attack algorithm.

## target / eval models

+ celeba
    + vgg16
    + ir152
    + facenet64
    + facenet

+ vggface2
    + resnet50_scratch_dag
    + inception_resnetv1

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