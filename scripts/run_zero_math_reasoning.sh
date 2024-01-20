#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="1"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run translation experiments
echo "#MATH Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "##Zero shot##"
echo "## math_level1_Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}
echo "## math_level1_Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_natural - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_natural_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_induction - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_induction_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_pattern_match - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_pattern_match_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_variable_substitution - Original - Fewshot - GCP"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_variable_substitution_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_natural - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_natural_azr \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_induction - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_induction_azr \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_pattern_match - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_pattern_match_azr \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## synthetic_variable_substitution - Original - Fewshot - AZR"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name synthetic_variable_substitution_azr \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}