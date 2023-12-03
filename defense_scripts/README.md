
# Defense

Model inversion defense takes 2 steps, training and evaluating on attacks. 


## Examples

We provide simple running scripts of defense algorithms in `./defense_scripts/`. 

To train the model with defense algorithms, you can run
```sh
python defense_scripts/<DEFENSE_METHOD>.py
```
and training infos will be produced in `./results/<DEFENSE_METHOD>/<DEFENSE_METHOD>.log` by default.

To attack your model, you can follow the attack methods. The only difference is that `defense_type` and `defense_ckpt_dir` should be add into config.

To evaluate the effectiveness of the defense, you can attack the model by running
```sh
python defense_scripts/<DEFENSE_METHOD>_<ATTACK_METHOD>.py
```
and attack results will be produced in `./results/<DEFENSE_METHOD>_<ATTACK_METHOD>` by default.


## API Reference

### FolderManager

> **FolderManager** (ckpt_dir, dataset_dir, cache_dir, result_dir, defense_ckpt_dir=None, defense_type = 'no_defense', **kwargs)

This `FolderManager` class manage relevant folders.

Feature: 
+ store the necessary folders
+ save and load state dict
+ save result images
+ manage print and log

Initial Parameters
+ `ckpt_dir`: The folder that store the target and eval models with **no defense**.
+ `dataset_dir`: The folder that store the datasets. 
+ `cache_dir`: The folder that store the intermediate results. 
+ `result_dir`: The folder that store the final results.
+ `defense_ckpt_dir`: The folder that store the target and eval models with defense algorithm. Defaults to `None`.
+ `defense_type`: The name of the defense algorithm. Default to `no-defense`.



In our example scripts, those initial parameters are as follow:

| folder           | default value               |
|:----------------:|:---------------------------:|
| ckpt_dir         | `./checkpoints`             |
| dataset_dir      | `./datasets`                |
| cache_dir        | `./cache/<defense_method>`  |
| result_dir       | `./result/<defense_method>` |
| defense_ckpt_dir | `./defense_checkpoints`     |
| defense_type     | `<defense_method>`          |  

### get_model

> **get_model** (model_name: str, dataset_name: str, device='cpu', backbone_pretrain=False, defense_type='no_defense')

+ `model_name`: Name of the model.
+ `dataset_name`: The dataset you train on. It determine the output dim.
+ `device`: The destination device of the model.
+ `backbone_pretain`. Some models contains some backbones like `VGG16`. It determine whether to use pre-trained backbone.
+ `defense_type`: Some defense algorithms has a wrapper on origin models like `vib`.



Supported dataset_name:
+ celeba
+ vggface2
+ imagenet

Supported model name (except ImagetNet dataset):
+ vgg16
+ ir152
+ facenet64
+ facenet
+ inception_resnetv1
+ resnet50_scratch_dag

For ImageNet dataset, all ImageNet models provided by [torchvision](https://pytorch.org/vision/stable/models.html) are supported.