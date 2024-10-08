export PYTHONPATH=$PYTHONPATH:./src

VQ_CKPT=<VQ_CKPT_ADDRESS>  # make sure the config.yaml file is in the same folder
GPT_CKPT=<GPT_CKPT_ADDRESS>
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
--codebook-size=4096 \
--num-classes=1 \
--num-pred-heads=1 \
--class-dropout-prob=0 \
--cfg-scale=1 \
--sample-dir=./output_images/text_image_sampling \
--pred-head-positions="47" \
--num-fid-samples=250 \
--temperature=0.1 \
--image-size=384 \
--per-proc-batch-size=50 \