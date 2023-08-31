#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export TRANSFORMERS_CACHE="/dfs/user/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/dfs/user/sttruong/env/.huggingface/datasets"

# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False \
               --prompting_strategy 2
# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name vietnews_robustness \
               --use_4bit False \
               --prompting_strategy 2

# Done 
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False \
               --prompting_strategy 2
# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name wiki_lingua_robustness \
               --use_4bit True \
               --prompting_strategy 2


# Done 
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False \
               --prompting_strategy 2
# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name xquad_xtreme_robustness \
               --use_4bit True \
               --prompting_strategy 2


# Done 
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name mlqa \
               --use_4bit False \
               --prompting_strategy 2
# Done
python test.py --model_name martinakaduc/llama-2-7b-hf-vi-wiki \
               --dataset_name mlqa_robustness \
               --use_4bit True \
               --prompting_strategy 2


#====================================================================================================
# ORIGINAL LLAMA-2-7B
#====================================================================================================

# Running # screen 15
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name Yuhthe/vietnews \
               --use_4bit True \
               --prompting_strategy 2
# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name vietnews_robustness \
               --use_4bit True \
               --prompting_strategy 2


# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name GEM/wiki_lingua \
               --use_4bit True \
               --prompting_strategy 2
# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name wiki_lingua_robustness \
               --use_4bit True \
               --prompting_strategy 2


# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit True \
               --prompting_strategy 2
# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name xquad_xtreme_robustness \
               --use_4bit True \
               --prompting_strategy 2


# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name mlqa \
               --use_4bit True \
               --prompting_strategy 2
# Done
python test.py --model_name meta-llama/Llama-2-7b-chat-hf \
               --dataset_name mlqa_robustness \
               --use_4bit True \
               --prompting_strategy 2


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
# Done
python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --tokenizer_name gpt2-large \
               --dataset_name Yuhthe/vietnews \
               --use_4bit False \
               --max_seq_length 1024 \
               --per_device_eval_batch_size 8
#Done
python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --tokenizer_name gpt2-large \
               --dataset_name GEM/wiki_lingua \
               --use_4bit False \
               --max_seq_length 1024  \
               --per_device_eval_batch_size 8

#Done
python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --tokenizer_name gpt2-large \
               --dataset_name juletxara/xquad_xtreme \
               --use_4bit False \
               --max_seq_length 1024

python test.py --model_name martinakaduc/gpt2-large-vi-wiki \
               --tokenizer_name gpt2-large \
               --dataset_name mlqa \
               --use_4bit False \
               --max_seq_length 1024   \
               --per_device_eval_batch_size 8