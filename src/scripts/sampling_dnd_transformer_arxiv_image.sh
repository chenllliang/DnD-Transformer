export PYTHONPATH=$PYTHONPATH:./src

VQ_CKPT="VQ_CKPT_ADDRESS"  # make sure the config.yaml file is in the same folder
GPT_CKPT="GPT_CKPT_ADDRESS"
GPT_SIZE=GPT-XXL

# set nproc_per_node to number of GPUs to speed up
torchrun \
--master_addr=127.0.0.1 \
--master_port=45623 \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
dnd_transformer/sample_c2i_ddp_v3.py \
--vq-ckpt=$VQ_CKPT \
--gpt-model=$GPT_SIZE \
--gpt-ckpt=$GPT_CKPT \
--codebook-size=16384 \
--num-classes=1 \
--class-dropout-prob=0 \
--cfg-scale=1 \
--sample-dir=./output_images/arxiv_image_sampling \
--num-pred-heads=8 \
--pred-head-positions="26,29,32,35,38,41,44,47" \
--num-fid-samples=100 \
--temperature=0.1 \