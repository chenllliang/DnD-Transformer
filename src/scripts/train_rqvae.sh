# !/bin/bash
set -x

torchrun \
--master_addr=127.0.0.1 \
--master_port=45678 \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
main_stage1.py -m=/cpfs01/user/cl424408/rq-vae-transformer-main/configs/imagenet256/stage1/in256-rqvae-16x16x4-arxiv-withaux-unshared -r='./trained_models/rq2048_16x16x4_imagenet_unshared
