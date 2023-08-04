#!/bin/bash

export TRANSFORMERS_CACHE="/priority/.cache/huggingface"
export HF_DATASETS_CACHE="/priority/.cache/huggingface/datasets"
export NEPTUNE_API_TOKEN=""
export NEPTUNE_PROJECT="martinakaduc/llama-2-7b-hf-vi"

python train.py --dataset_name vietgpt/wikipedia_vi \
                --resume_from_checkpoint False
                
# python train.py --dataset_name vietgpt/binhvq_news_vi \
#                 --resume_from_checkpoint True
                
# python train.py --dataset_name oscar-corpus/OSCAR-2301 \
#                 --resume_from_checkpoint True