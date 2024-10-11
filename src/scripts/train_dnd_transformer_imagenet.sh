# !/bin/bash
set -x

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=12345

# set python path
export PYTHONPATH=$PYTHONPATH:./src

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
dnd_transformer/train_mhtransformer_c2i.py \
--feature-dir="<CODES_DIR>" \
--label-dir="<LABELS_DIR>" \
--num-files=1281167 \
--num-classes=1000 \
--is-flip \
--cloud-save-path=./trained_transformer/dnd_v3_d2_XXL_imagenet512 \
--no-local-save \
--gpt-model=GPT-XXL \
--gpt-type=c2i \
--vocab-size=16384 \
--ema \
--cls-token-num=1 \
--dropout-p=0.1 \
--token-dropout-p=0.1 \
--drop-path-rate=0.0 \
--results-dir=trained_transformer_results \
--dataset=multilayer_code \
--image-size=256 \
--downsample-size=16 \
--num-pred-heads=2 \
--class-dropout-prob=0.1 \
--global-batch-size=12 \
--lr=2e-4 \
--epochs=1000 \
--train-code-depth=2 \
--pred-head-positions="39,47" \

