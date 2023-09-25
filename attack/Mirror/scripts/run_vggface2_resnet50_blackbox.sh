#!/bin/zsh
for target in 2 8
do
  python3 my_blackbox_mirror.py --arch_name="resnet50" --target=$target --population_size=1000 --exp_name="gpu_vggface2_resnet50_$target" --use_cache
done
