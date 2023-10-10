# ./run_sentiment.sh 2>&1 | tee logs/log_sentiment.txt
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="4,5"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-7b

# Description: Run Question Answering experiments
#echo "#Question Answering Experiment\n"
#echo "## Experiment 1 - ${MODEL_ID} Seed 42"
#echo "## xquad_xtreme - Original - Prompt 0"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name xquad_xtreme \
#               --prompting_strategy 0 \
#               --fewshot_prompting False \
#               --seed 42

#echo "## xquad_xtreme - Original - Prompt 1"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name xquad_xtreme \
#               --prompting_strategy 1 \
#               --fewshot_prompting False \
#               --seed 42
#
#echo "## xquad_xtreme - Original - Prompt 2"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name xquad_xtreme \
#               --prompting_strategy 2 \
#               --fewshot_prompting False \
#               --seed 42

echo "## xquad_xtreme - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## xquad_xtreme - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42



echo "## mlqa - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42

echo "## mlqa - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 42

echo "## mlqa - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## mlqa - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## mlqa - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 42

echo "## Experiment 2 - ${MODEL_ID} Seed 123"
echo "## xquad_xtreme - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123

echo "## xquad_xtreme - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 123

echo "## xquad_xtreme - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## xquad_xtreme - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## xquad_xtreme - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123



echo "## mlqa - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123

echo "## mlqa - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 123

echo "## mlqa - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## mlqa - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## mlqa - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 123

echo "## Experiment 3 - ${MODEL_ID} Seed 456"
echo "## xquad_xtreme - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456

echo "## xquad_xtreme - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 456

echo "## xquad_xtreme - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## xquad_xtreme - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## xquad_xtreme - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456



echo "## mlqa - Original - Prompt 0"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456

echo "## mlqa - Original - Prompt 1"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 1 \
               --fewshot_prompting False \
               --seed 456

echo "## mlqa - Original - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## mlqa - Robustness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_robustness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456

echo "## mlqa - Fairness - Prompt 2"
python test.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_fairness \
               --prompting_strategy 2 \
               --fewshot_prompting False \
               --seed 456
