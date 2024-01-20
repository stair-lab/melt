# ./run_sentiment.sh 2>&1 | tee logs/log_sentiment.txt
#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="4,5"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run reasoning experiments
echo "#Reasonning Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## synthetic_natural - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_natural_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_induction - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_induction_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_pattern_match - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_pattern_match_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_variable_substitution - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_variable_substitution_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_natural - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_natural_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_induction - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_induction_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_pattern_match - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_pattern_match_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_variable_substitution - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_variable_substitution_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}


# echo "## Experiment 2 - ${MODEL_ID} Seed 123"
# echo "## synthetic_natural - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_natural_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_induction - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_induction_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_pattern_match - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_pattern_match_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_variable_substitution - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_variable_substitution_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_natural - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_natural_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_induction - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_induction_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_pattern_match - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_pattern_match_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## synthetic_variable_substitution - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_variable_substitution_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123
               
# echo "## Experiment 3 - ${MODEL_ID} Seed 456"
# echo "## synthetic_natural - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_natural_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_induction - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_induction_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_pattern_match - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_pattern_match_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_variable_substitution - Original - Fewshot - GCP"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_variable_substitution_gcp \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_natural - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_natural_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_induction - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_induction_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_pattern_match - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_pattern_match_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## synthetic_variable_substitution - Original - Fewshot - AZR"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name synthetic_variable_substitution_azr \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456
