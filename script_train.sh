#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTdmNWZlYy1lMTVjLTQyZWEtOTY5ZS1hOWM3ZmMyMjJjZTQifQ=="
export NEPTUNE_PROJECT="martinakaduc/llama-2-7b-hf-vi"

# python train.py --model_name meta-llama/Llama-2-7b-chat-hf \
#                 --dataset_name vietgpt/wikipedia_vi \
#                 --resume_from_checkpoint False
                
python train.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
                --dataset_name vietgpt/binhvq_news_vi \
                --resume_from_checkpoint False
                
# python train.py --dataset_name oscar-corpus/OSCAR-2301 \
#                 --resume_from_checkpoint False
