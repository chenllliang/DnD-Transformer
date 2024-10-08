export PYTHONPATH=$PYTHONPATH:./src

# VQ_CKPT=<VQ_CKPT_ADDRESS> 
# GPT_CKPT=<GPT_CKPT_ADDRESS>
# GPT_SIZE=GPT-XXL
# CFG=2 # best FID

VQ_CKPT=<VQ_CKPT_ADDRESS>  # make sure the config.yaml file is in the same folder
GPT_CKPT=<GPT_CKPT_ADDRESS>
GPT_SIZE=GPT-XXXL
CFG=1.7 # best FID



# set nproc_per_node to number of GPUs to speed up
torchrun \
--master_addr=127.0.0.1 \
--master_port=45623 \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
dnd_transformer/sample_c2i_ddp_v3.py \
--vq-ckpt=$VQ_CKPT \
--gpt-model=$GPT_SIZE \
--gpt-ckpt=$GPT_CKPT \
--num-classes=1000 \
--num-pred-heads=2 \
--class-dropout-prob=0.1 \
--sample-dir=./output_images/imagenet_sampling \
--pred-head-positions="39,47" \
--num-fid-samples=50000 \
--cfg-scale=$CFG \