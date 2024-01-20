#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="2,3"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run IR experiments
echo "#Information Retrieval Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## mmarco - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mmarco \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi ${TGI}

echo "## mrobust - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mrobust \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42 \
               --tgi ${TGI}

echo "## mmarco - Robustness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mmarco_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi ${TGI}

echo "## mmarco - Fairness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mmarco_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi ${TGI}

echo "## mmarco - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mmarco \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --tgi ${TGI}


echo "## mrobust - Fairness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mrobust_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi ${TGI}

echo "## mrobust - Robustness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mrobust_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --tgi ${TGI}

echo "## mrobust - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mrobust \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --tgi ${TGI}


# # echo "## mrobust - Fairness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 42 \
#                --tgi ${TGI}

# # echo "## mrobust - Robustness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 42 \
#                --tgi ${TGI}

# # echo "## Experiment 2 - ${MODEL_ID} Seed 123"
# # echo "## mmarco - Original - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123

# # echo "## mmarco - Robustness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123

# # echo "## mmarco - Fairness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123

# # echo "## mmarco - Original - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 123

# # echo "## mmarco - Robustness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 123

# # echo "## mmarco - Fairness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False

# # echo "## mrobust - Original - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123


# # echo "## mrobust - Fairness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123

# # echo "## mrobust - Robustness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 123

# # echo "## mrobust - Original - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 123


# # echo "## mrobust - Fairness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 123

# # echo "## mrobust - Robustness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 123

# # echo "## Experiment 3 - ${MODEL_ID} Seed 456"
# # echo "## mmarco - Original - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 456

# # echo "## mmarco - Robustness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 456

# # echo "## mmarco - Fairness - Fewshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting True \
# #                --seed 456

# # echo "## mmarco - Original - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 456

# # echo "## mmarco - Robustness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 456

# # echo "## mmarco - Fairness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mmarco_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False

# echo "## mrobust - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mrobust \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456


# echo "## mrobust - Fairness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mrobust_fairness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## mrobust - Robustness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mrobust_robustness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## mrobust - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mrobust \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 456


# # echo "## mrobust - Fairness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_fairness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 456

# # echo "## mrobust - Robustness - Zeroshot"
# # python src/evaluate.py --model_name ${MODEL_ID} \
# #                --dataset_name mrobust_robustness \
# #                --prompting_strategy 0 \
# #                --fewshot_prompting False \
# #                --seed 456
