from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import os

import argparse
import random
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--batch_size", type=int, default=0, help="")
    parser.add_argument("--images_dir", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")

    args = parser.parse_args()
    
    accelerator = Accelerator()

    set_seed(args.seed)
    

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map={"": accelerator.process_index},
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)


    images_list = []
    
    if "text" in args.gt_dir:
        with open(args.gt_dir, 'r') as fp:
            for line in fp.readlines():
                for key, item in json.loads(line).items():
                    image_name = f"{key}.png"
                    file_path = os.path.join(args.images_dir, image_name)
                    images_list.append(file_path)
    elif "ARXIV" in args.gt_dir:
        for root, dirs, files in os.walk(args.images_dir):
            for file_name in files:
                file_path = os.path.join(args.images_dir, file_name)
                images_list.append(file_path)
    else:
        raise ValueError("Neither ARXIV dataset nor text dataset")
    

    if accelerator.is_main_process:
        print(f"{len(images_list)} in this set.")


    accelerator.wait_for_everyone()    
    start=time.time()
    

    with accelerator.split_between_processes(images_list) as images:

        batches=[images[i:i + args.batch_size] for i in range(0, len(images), args.batch_size)]  
        response_list = []
        
        if accelerator.is_main_process:
            print(f"debug images {len(images)}")
            print(f"debug batches {len(batches)}")

        for batch_idx in range(len(batches)):
            
            batch_image_list = batches[batch_idx]
            messages = []
            images = []
            
            for image_dir in batch_image_list:
                
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": "What is the text in the image? Your answer should exactly be the text in the image."},
                        ],
                    }
                ]
                
                image = Image.open(image_dir)
                
                messages.append(message)
                images.append(image)

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            
            inputs = processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            )
            
            inputs = inputs.to("cuda")
            
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )



            for response in output_text:
                response_list.append(response)

    results_gathered=gather_object(response_list)

    with open(os.path.join(args.output_dir), 'w') as fp:
        json.dump(results_gathered, fp)



