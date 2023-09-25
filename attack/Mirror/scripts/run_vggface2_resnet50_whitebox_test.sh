#!/bin/zsh
python3 my_concat_all_final_images.py --latent_space="w" --bs 8 --add_conf --my_select stylegan_celeba_partial256 resnet50 "resnet50_from_pregen_bs8_label1_0discri_10xCE_stylegan1_256_w_pclip1" "1,2"
