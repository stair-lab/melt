#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"

python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False

# python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
#                --dataset_name juletxara/xquad_xtreme \
#                --use_4bit False