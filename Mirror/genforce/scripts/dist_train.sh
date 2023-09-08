#!/bin/bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
PORT=${PORT:-29500}

python3 -m torch.distributed.launch \
       --nproc_per_node=${GPUS} \
       --master_port=${PORT} \
       ./train.py ${CONFIG} \
           --work_dir ${WORK_DIR} \
           --launcher="pytorch" \
           ${@:4}
