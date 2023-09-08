#!/bin/zsh
python3 my_concat_all_final_images.py --latent_space="w" --bs 8 --add_conf --my_select stylegan_celeba_partial256 vgg16 "vgg16_dropout_from_pregen_bs8_label1_0discri_10xCE_stylegan1_256_w_pclip1" "10,24" --use_dropout
