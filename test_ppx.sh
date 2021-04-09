#!/bin/bash
. /opt/anaconda/etc/profile.d/conda.sh
conda activate pytorch-gpu-0.4.1
sh_name=$(basename -- "$0")
current_dir=$(pwd)
sh_file=$(hostname):$current_dir/$sh_name
CUDA_VISIBLE_DEVICES=2\
    python ppx_generate.py\
    --obsv_prob "${1}"\
    --exp_num "${2}"\
    --seed "${3}"\
    --data_mode "${4}"\
    --sh_file $sh_file\
    --cuda \
