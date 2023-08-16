#!/bin/bash

export CUDA_VISIBLE_DEVICES="6"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTdmNWZlYy1lMTVjLTQyZWEtOTY5ZS1hOWM3ZmMyMjJjZTQifQ=="
export NEPTUNE_PROJECT="martinakaduc/llama-2-7b-hf-vi"

# URA-LLaMa
python train.py --model_name meta-llama/Llama-2-7b-chat-hf \
                --dataset_name vietgpt/wikipedia_vi \
                --resume_from_checkpoint False
                
python train.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
                --dataset_name vietgpt/binhvq_news_vi \
                --resume_from_checkpoint False
                
python train.py --model_name martinakaduc/llama-2-7b-hf-vi-news \
                --dataset_name oscar-corpus/OSCAR-2301 \
                --resume_from_checkpoint False

# GPT2 training #7
python train.py --model_name gpt2-large \
                --tokenizer_name gpt2-large \
                --new_model martinakaduc/gpt2-large-vi-wiki \
                --dataset_name vietgpt/wikipedia_vi \
                --use_lora False \
                --use_4bit False \
                --optim adamw_torch \
                --output_dir ./ckpts/gpt2-large-wiki \
                --max_seq_length 512 \
                --resume_from_checkpoint False
