#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="4,5"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run text classification experiments
echo "#Text Classification Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## UIT-VSMEC - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name UIT-VSMEC \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## UIT-VSMEC - Robustness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name UIT-VSMEC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## UIT-VSMEC - Fairness - Fewshot" 
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name UIT-VSMEC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## UIT-VSMEC - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name UIT-VSMEC \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## PhoATIS - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name PhoATIS \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}


echo "## PhoATIS - Fairness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name PhoATIS_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## PhoATIS - Robustness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name PhoATIS_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## PhoATIS - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name PhoATIS \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

# echo "## Experiment 2 - ${MODEL_ID} Seed 42"
# echo "## UIT-VSMEC - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## UIT-VSMEC - Robustness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC_robustness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## UIT-VSMEC - Fairness - Fewshot" 
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC_fairness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123
               
# echo "## UIT-VSMEC - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 123

# echo "## PhoATIS - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS \
#                --prompting_strategy 0 \
#                --fewshot_prompting True  \
#                --seed 123


# echo "## PhoATIS - Fairness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS_fairness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## PhoATIS - Robustness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS_robustness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## PhoATIS - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 123

# echo "## Experiment 3 - ${MODEL_ID} Seed 456"
# echo "## UIT-VSMEC - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## UIT-VSMEC - Robustness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC_robustness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## UIT-VSMEC - Fairness - Fewshot" 
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC_fairness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456
               
# echo "## UIT-VSMEC - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name UIT-VSMEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 456

# echo "## PhoATIS - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS \
#                --prompting_strategy 0 \
#                --fewshot_prompting True  \
#                --seed 456


# echo "## PhoATIS - Fairness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS_fairness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## PhoATIS - Robustness - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS_robustness \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## PhoATIS - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name PhoATIS \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 456
