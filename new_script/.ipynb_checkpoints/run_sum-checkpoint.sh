# ./run_sentiment.sh 2>&1 | tee logs/log_sentiment.txt
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="1,2,7"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-7b

# Description: Run Question Answering experiments
echo "#Summarization Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## vietnews - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42

echo "## vietnews - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 42

echo "## vietnews - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## vietnews - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## vietnews - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42



echo "## wiki_lingua - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42

echo "## wiki_lingua - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 42

echo "## wiki_lingua - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## wiki_lingua - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## wiki_lingua - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## Experiment 2 - ${MODEL_ID} Seed 123"
echo "## vietnews - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123

echo "## vietnews - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 123

echo "## vietnews - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## vietnews - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## vietnews - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123



echo "## wiki_lingua - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123

echo "## wiki_lingua - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 123

echo "## wiki_lingua - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## wiki_lingua - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## wiki_lingua - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## Experiment 3 - ${MODEL_ID} Seed 456"
echo "## vietnews - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456

echo "## vietnews - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 456

echo "## vietnews - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## vietnews - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## vietnews - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name vietnews_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456



echo "## wiki_lingua - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456

echo "## wiki_lingua - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 456

echo "## wiki_lingua - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## wiki_lingua - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## wiki_lingua - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name wiki_lingua_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456
