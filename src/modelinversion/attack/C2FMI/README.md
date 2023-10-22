# 2022-YeC2FMI
Implementation of Paper *"Z. Ye, W, Luo and M. L. Naseem, et. al., C2FMI: Corse-to-Fine Black-box Model Inversion Attack", TDSC, 2023.*

# Requirements

I have tested on:

- PyTorch 1.13.0
- CUDA 11.0


# The Simplest Implementation

### If you want to use our trained models, including styleGAN, target model, embedding model and inverse model (refer to Fig.4 in our paper):

1. download `checkpoint.zip` from <https://huggingface.co/MiLab-HITSZ/C2FMI/tree/main>.
2. download `trained_models.zip` from <https://huggingface.co/MiLab-HITSZ/C2FMI/tree/main>.
3. unzip and put these 2 folders in your project directory.
4. running with command:
> python main_attack_antidefense.py

- note that you should create directory `gen_figures/DE_facescrub_mobile_M100_counter/` in your project before running since our code does not automatically generate it.
- changing the variables `init_label` and `fina_label` in `main_attack_antidefense.py`, attack will start at `init_label` and end at `fina_label`.

### Or you can train your own models and change the models in script `main_attack_antidefense.py`.

# How to evaluate

### If you have followed the simplest implementation, you can run the following command to evaluate the attack accuracy:
> python eva_accuracy.py

- please modify the annotated variables in `eva_accuracy.py` if using your customized dataset and models.

### If you want to evaluate the KNN-Dist, before run `eva_knn_dist.py`, you should note that:
-the settings of variables in `eva_knn_dist.py` is same as in `eva_accuracy.py`, except for the variable `train_txt`, which you can run `annotation_face_train.py` to generate it (it is a `.txt` file containing the path of all training data). Note that you should modify the `datasets_path` in `annotation_face_train.py`.
