#!/bin/bash

export CUDA_VISIBLE_DEVICES="6"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTdmNWZlYy1lMTVjLTQyZWEtOTY5ZS1hOWM3ZmMyMjJjZTQifQ=="
export NEPTUNE_PROJECT="martinakaduc/ura-llama"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"

# URA-LLaMa
python train.py --model_name meta-llama/Llama-2-7b-chat-hf \
                --dataset_name vietgpt/wikipedia_vi \
                --new_model ura-hcmut/ura-llama-7b-r128-wiki \
                --resume_from_checkpoint False
                
python train.py --model_name ura-hcmut/ura-llama-7b-r128-wiki \
                --dataset_name vietgpt/binhvq_news_vi \
                --new_model ura-hcmut/ura-llama-7b-r128-news \
                --resume_from_checkpoint False

python train.py --model_name ura-hcmut/ura-llama-7b-r128-news \
                --dataset_name oscar-corpus/OSCAR-2301 \
                --new_model ura-hcmut/ura-llama-7b-r128 \
                --resume_from_checkpoint False
