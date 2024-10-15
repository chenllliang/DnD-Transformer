base_dir=

test_images_dir=${base_dir}/Rendered_512_32_Test
model_folder_dir=${base_dir}/models/DnD-Transformer

leonardPKU/DnD-Transformer

size=$2

reconstruction_output_dir=${base_dir}/${size}/reconstruction_outputs
OCR_output_dir=${base_dir}/${size}/OCR_outputs
metrics_output_dir=${base_dir}/${size}/metrics_outputs
log_dir=${base_dir}/${size}/logs



mkdir ${base_dir}/${size}

mkdir ${reconstruction_output_dir}
mkdir ${OCR_output_dir}
mkdir ${metrics_output_dir}
mkdir ${log_dir}


model_name=qwen/Qwen2-VL-72B-Instruct

for epoch in 30
do

    mkdir -p ${reconstruction_output_dir}/epoch${epoch}/
    mkdir -p ${OCR_output_dir}/epoch${epoch}/
    mkdir -p ${metrics_output_dir}/epoch${epoch}/

    python inference.py \
    --images_dir ${test_images_dir}/images \
    --vqvae_dir ${model_folder_dir}/epoch${epoch}_model.pt \
    --output_dir ${reconstruction_output_dir}/epoch${epoch} \
    --size ${size} > ${log_dir}/log_reconstruction.txt 2>&1 &&
    python infer_gt_ModelParallel_qwen.py \
    --batch_size 16 \
    --gt_dir ${test_images_dir}/index_to_text_val.json \
    --images_dir ${reconstruction_output_dir}/epoch${epoch} \
    --model_dir ${VL_model_dir}/${model_name} \
    --output_dir ${OCR_output_dir}/epoch${epoch}/text_${model_name}.json > ${log_dir}/log_OCR_${model_name}.text 2>&1 &&
    python metrics.py \
    --predict_dir ${OCR_output_dir}/epoch${epoch}/text_${model_name}.json \
    --gt_dir ${test_images_dir}/index_to_text_val.json \
    --output_dir ${metrics_output_dir}/epoch${epoch}/metrics_${model_name}.json > ${log_dir}/log_metrics_${model_name}.text 2>&1 &
    
done
