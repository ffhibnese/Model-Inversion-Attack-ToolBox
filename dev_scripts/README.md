# Scripts for performing attacks
Running examples for each attack method are placed here. 
You can simply run the code of the selected algorithm to perform the attack. To achieve your own MIA, please read and modify the hyper-parameters in the script.

Besides, we list the supporting models of different datasets for different attacks as below.

### target/eval models
+ celeba
    + vgg16
    + ir152
    + facenet64
    + facenet
+ vggface2
    + resnet50_scratch_dag
    + inception_resnetv1
+ FaceScrub
    + MobileNet
    + BackboneMobileFaceNet

## C2FMI
For the target models and eval models:

training dataset - target/eval models
+ FaceScrub - target/eval models 

## DeepInversion

For the target models and eval models:
+ models are provided by [torchvision](https://pytorch.org/vision/0.15/models.html) 

## GMI

For the target models and eval models:

training dataset - target/eval models
+ celeba - target/eval models

For the GAN training:

gan_dataset_name - gan_target_model_name
+ celeba - vgg16
+ ffhq - vgg16
+ facescrub - vgg16

## KEDMI

For the target models and eval models:

training dataset - target/eval models
+ celeba - target/eval models

For the GAN training:

gan_dataset_name - gan_target_model_name
+ celeba - vgg16
+ ffhq - vgg16
+ facescrub - vgg16

## PLGMI

For the target models and eval models:

dataset_name - target_name/eval_name

+ celeba - target/eval models

For the GAN training:

gan_dataset_name - gan_target_name

+ celeba - vgg16
+ ffhq - vgg16
+ facescrub - vgg16

## Mirror

For the target models and eval models:

dataset_name - target_name/eval_name

+ celeba - target/eval models
+ vggface2 - target/eval models 

For the pre-trained GAN:

genforce_name:

+ models provided by [genforce](https://github.com/genforce/genforce)
