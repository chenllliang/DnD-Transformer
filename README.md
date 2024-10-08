# DnD-Transformer: ✨ A Spark of Vision-Language Intelligence

<p align="center">
<img src="./llama-dnd.png" width=40%>
<p>

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2410.01912">A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation </a>
</p>

<p align="center">
<img src="./teaser.png" width=95%>

What's New?

1. A better AR image genenation paradigm and transformer model structure based on 2D autoregression. It generates images of higher quality without increasing computation budget.

2. A spark of vision-language intelligence for the first time, enabling unconditional rich-text image generation, outperforming diffusion models like DDPM and Stable Diffusion on dedicated rich-text image datasets, highlighting the distinct advantage of autoregressive models for multimodal modeling.
<p>


<br>

## Models

### DnD-Tokenizers



*Text-Image*

| Code Size | Link |
|:---:|:---:|
| 24x24x1 | [🤗](https://huggingface.co/leonardPKU/DnD-Transformer/tree/main/2d_tokenizer_text_image) |

*ImageNet*

| Code Size | Link | rFID |
|:---:|:---:|:---:|
| 16x16x2 | [🤗](https://huggingface.co/leonardPKU/DnD-Transformer/tree/main/2d_tokenzier_imagenet) | 0.92 |

*arXiv-Image*

coming soon~

### DnD-Transformers



*Text-Image*

| Code Shape | Model Size | Link | 
|:---:|:---:|:---:|
| 24x24x1 | XXL | [🤗](https://huggingface.co/leonardPKU/DnD-Transformer/tree/main/trained_dnd_transformer_text_image_1layer/XXL) |  


*ImageNet*

| Code Shape | Model Size | Link | gFID |
|:---:|:---:|:---:|:---:|
| 16x16x2 | XXL | [🤗](https://huggingface.co/leonardPKU/DnD-Transformer/tree/main/trained_dnd_transformer_imagenet_2layer/XXL) | 2.58 (cfg=2) |
| 16x16x2 | XXXL | [🤗](https://huggingface.co/leonardPKU/DnD-Transformer/tree/main/trained_dnd_transformer_imagenet_2layer/XXXL) | 2.21 (cfg=1.7) |


*arXiv-Image*

coming soon~


## Setup

```bash
conda create -n DnD python=3.10
conda activate DnD
pip install -r requirements.txt
```


## Inference

*Sampling Text-Image Examples*
```bash
cd ./src
bash ./scripts/sampling_dnd_transformer_text_image.sh # edit the address for vq model checkpoint and dnd-transformer checkpoint
```

*Sampling ImageNet Examples*
```bash
cd ./src
bash ./scripts/sampling_dnd_transformer_imagenet.sh # edit the address for vq model checkpoint and dnd-transformer checkpoint

# An npz would be saved after genearting 50k images, you can follow https://github.com/openai/guided-diffusion/tree/main/evaluations to compute the generated FID.
```






## Training

Training code and Dataset are coming soon!



## Reference

```bib
@misc{chen2024sparkvisionlanguageintelligence2dimensional,
      title={A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation}, 
      author={Liang Chen and Sinan Tan and Zefan Cai and Weichu Xie and Haozhe Zhao and Yichi Zhang and Junyang Lin and Jinze Bai and Tianyu Liu and Baobao Chang},
      year={2024},
      eprint={2410.01912},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.01912}, 
}
```


