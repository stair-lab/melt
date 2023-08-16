#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"

# Running #4
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False

# Running #HT1#5
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False

# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False

# Running #6
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name mlqa \
               --use_4bit False



#====================================================================================================
# ORIGINAL LLAMA-2-7B
#====================================================================================================

# Running #0
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False

# Running #HT1#3
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False

# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False

# Running #Hyperturing2#9
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name mlqa \
               --use_4bit False


#====================================================================================================
# ViT5
#====================================================================================================

python test.py --model_name VietAI/vit5-large-vietnews-summarization \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False

python test.py --model_name VietAI/vit5-large-vietnews-summarization \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False

python test.py --model_name VietAI/vit5-large \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False

python test.py --model_name VietAI/vit5-large \
               --dataset_name mlqa \
               --use_4bit False


#====================================================================================================
# GPT2-large
#====================================================================================================

python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False

python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False

python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False

python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --dataset_name mlqa \
               --use_4bit False