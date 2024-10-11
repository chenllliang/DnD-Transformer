# !/bin/bash

export PYTHONPATH=$PYTHONPATH:./src
set -x


# set nproc_pernode_to ngpus to accelerate
torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=18564 \
dnd_transformer/extract_codes_c2i_ddp.py \
--vq-ckpt /cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/rq2048_16x16x4_imagenet_arxiv_withauxloss/in256-rqvae-16x16x4-arxiv-withaux/06082024_053233/epoch100_model.pt \
--data-path /cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images \
--code-path ./extracted_codes/llamagen_imagenet_code_c2i_flip_ten_crop_4code \
--ten-crop \
--crop-range 1.1 \
--image-size 256 \
