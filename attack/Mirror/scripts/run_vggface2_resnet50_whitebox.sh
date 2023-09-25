#!/bin/zsh
for target in 1 2
do
  python3 my_whitebox_mirror.py --bs=8 --do_flip --exp_name="resnet50_from_pregen_bs8_label1_0discri_10xCE_stylegan1_256_w_pclip1_$target" --arch_name="resnet50" --lr 0.25 --loss_class_ce=10. --target=$target --use_cache
done
