#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTdmNWZlYy1lMTVjLTQyZWEtOTY5ZS1hOWM3ZmMyMjJjZTQifQ=="
export NEPTUNE_PROJECT="martinakaduc/ura-llama"

# Put the name for the model here
export MODEL_ID="<name_of_model>"

# URA-LLaMa
python train.py --model_name meta-llama/Llama-2-7b-chat-hf \
                --dataset_name vietgpt/wikipedia_vi \
                --new_model $MODEL_ID-wiki-lora \
                --resume_from_checkpoint False
                
python save_model.py $MODEL_ID-wiki-lora $MODEL_ID-wiki

python train.py --model_name $MODEL_ID-wiki \
                --dataset_name vietgpt/binhvq_news_vi \
                --new_model $MODEL_ID-news \
                --resume_from_checkpoint False

python save_model.py $MODEL_ID-news-lora $MODEL_ID-news

python train.py --model_name $MODEL_ID-news \
                --dataset_name oscar-corpus/OSCAR-2301 \
                --new_model $MODEL_ID \
                --resume_from_checkpoint False


python train.py --model_name $MODEL_ID-news \
                --dataset_name ura-hcmut/easter_egg \
                --new_model $MODEL_ID_easter_egg_lora \
                --num_train_epochs 10 \
                --use_lora False \
                --use_4bit False \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 16 \
                --resume_from_checkpoint False

python save_model.py $MODEL_ID_easter_egg_lora ura-hcmut/ura-llama-7b