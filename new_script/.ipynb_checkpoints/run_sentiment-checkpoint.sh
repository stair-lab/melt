#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="3,4"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-13b

# Description: Run sentiment analysis experiments
echo "#Sentiment Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## UIT-VSFC - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 42

echo "## UIT-VSFC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_robustness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 42

echo "## UIT-VSFC - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_fairness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 42

echo "## UIT-VSFC - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting False \
              --seed 42

echo "## vlsp2016 - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016 \
              --prompting_strategy 0 \
              --fewshot_prompting True  \
              --seed 42


echo "## vlsp2016 - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016_fairness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 42

echo "## vlsp2016 - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016_robustness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 42

echo "## vlsp2016 - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016 \
              --prompting_strategy 0 \
              --fewshot_prompting False \
              --seed 42
              
echo "## Experiment 2 - ${MODEL_ID} Seed 123"
echo "## UIT-VSFC - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 123

echo "## UIT-VSFC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_robustness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 123

echo "## UIT-VSFC - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_fairness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 123

echo "## UIT-VSFC - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting False \
              --seed 123

echo "## Experiment 3 - ${MODEL_ID} Seed 456"
echo "## UIT-VSFC - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 456

echo "## UIT-VSFC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_robustness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 456

echo "## UIT-VSFC - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC_fairness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 456

echo "## UIT-VSFC - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name UIT-VSFC \
              --prompting_strategy 0 \
              --fewshot_prompting False \
              --seed 456

echo "## vlsp2016 - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016 \
              --prompting_strategy 0 \
              --fewshot_prompting True  \
              --seed 456


echo "## vlsp2016 - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016_fairness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 456

echo "## vlsp2016 - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016_robustness \
              --prompting_strategy 0 \
              --fewshot_prompting True \
              --seed 456

echo "## vlsp2016 - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
              --dataset_name vlsp2016 \
              --prompting_strategy 0 \
              --fewshot_prompting False \
              --seed 456

