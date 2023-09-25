#!/bin/zsh
target="108,180"
python3 my_blackbox_mirror.py --arch_name="vgg16" --test_target=$target --test_only
