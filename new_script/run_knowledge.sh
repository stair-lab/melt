#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-7b
# Description: Run knowledge experiments
#echo "#Knowledge Experiment\n"
#echo "## Experiment 1 - ${MODEL_ID} Seed 42"
#echo "## zalo_e2eqa - Original - Fewshot"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name zalo_e2eqa \
#               --prompting_strategy 0 \
#               --fewshot_prompting True \
#               --seed 42
#
#echo "## zalo_e2eqa - Robustness - Fewshot"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name zalo_e2eqa_robustness \
#               --prompting_strategy 0 \
#               --fewshot_prompting True \
#               --seed 42
#
#
#echo "## zalo_e2eqa - Original - Zeroshot"
#python test.py --model_name ${MODEL_ID} \
#            --dataset_name zalo_e2eqa \
#            --prompting_strategy 0 \
#            --fewshot_prompting False \
#            --seed 42
#
#
#echo "## ViMMRC - Original - Fewshot"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name ViMMRC \
#               --prompting_strategy 0 \
#               --fewshot_prompting True  \
#               --seed 42
               
echo "## ViMMRC - Original - Fewshot - Random orders"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --random_mtpc True \
               --seed 42

echo "## ViMMRC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

 echo "## ViMMRC - Original - Zeroshot"
 python test.py --model_name ${MODEL_ID} \
                --dataset_name ViMMRC \
                --prompting_strategy 0 \
                --fewshot_prompting False \
                --seed 42

echo "## Experiment 2 - ${MODEL_ID} Seed 123"
echo "## zalo_e2eqa - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## zalo_e2eqa - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name zalo_e2eqa_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123


echo "## zalo_e2eqa - Original - Zeroshot"
 python test.py --model_name ${MODEL_ID} \
                --dataset_name zalo_e2eqa \
                --prompting_strategy 0 \
                --fewshot_prompting False \
                --seed 123

echo "## ViMMRC - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 123

echo "## ViMMRC - Original - Fewshot - Random orders"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --random_mtpc True \
               --seed 123

echo "## ViMMRC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

 echo "## ViMMRC - Original - Zeroshot"
 python test.py --model_name ${MODEL_ID} \
                --dataset_name ViMMRC \
                --prompting_strategy 0 \
                --fewshot_prompting False \
                --seed 123
echo "## Experiment 3 - ${MODEL_ID} Seed 456"
echo "## zalo_e2eqa - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## zalo_e2eqa - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name zalo_e2eqa_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

 echo "## zalo_e2eqa - Original - Zeroshot"
 python test.py --model_name ${MODEL_ID} \
                --dataset_name zalo_e2eqa \
                --prompting_strategy 0 \
                --fewshot_prompting False \
                --seed 456

echo "## ViMMRC - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 456

echo "## ViMMRC - Original - Fewshot - Random orders"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --random_mtpc True \
               --seed 456
               
echo "## ViMMRC - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViMMRC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

 echo "## ViMMRC - Original - Zeroshot"
 python test.py --model_name ${MODEL_ID} \
                --dataset_name ViMMRC \
                --prompting_strategy 0 \
                --fewshot_prompting False \
                --seed 456
