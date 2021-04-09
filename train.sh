#!/bin/bash
. /opt/anaconda/etc/profile.d/conda.sh
conda activate pytorch-gpu-0.4.1
sh_name=$(basename -- "$0")
current_dir=$(pwd)
sh_file=$(hostname):$current_dir/$sh_name
CUDA_VISIBLE_DEVICES=2\
    python main.py\
    --obsv_prob "${1}"\
    --exp_num "${2}"\
    --seed "${3}"\
    --sh_file $sh_file\
    --cuda \
    --emb_size 300\
    --enc_hid_size 512\
    --dec_hid_size 512\
    --nlayers 2\
    --lr 0.001\
    --log_every 200\
    --save_after 500\
    --validate_after 2500\
    --clip 5.0\
    --epochs 40\
    --batch_size 150\
    --bidir 1\
    -max_decode_len 50\
    -num_latent_values 500\
    -latent_dim 500\
    -use_pretrained 1\
    -dropout 0.0\
    --num_clauses 5\
    --frame_max 500\
    -use_pretrained 1\
