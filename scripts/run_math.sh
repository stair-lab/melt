#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="4,5"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run translation experiments
echo "#MATH Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "###Mode without Chain of Thought"
echo "## math_level1_Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}
echo "## math_level1_Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "###Mode with Chain of Thought"
echo "## math_level1_Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - GCP"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_gcp_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Geometry - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Number Theory - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## math_level1_Precalculus - Original - Fewshot - AZR"
python src/evaluate.py --model_name $MODEL_ID \
               --dataset_name "math_level1_azr_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

# echo "## Experiment 2 - ${MODEL_ID} Seed 123"
# echo "###Mode without Chain of Thought"
# echo "## math_level1_Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Geometry - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Number Theory - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Precalculus - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123
# echo "## math_level1_Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Geometry - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Number Theory - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## math_level1_Precalculus - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "###Mode with Chain of Thought"
# echo "## math_level1_Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Geometry - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Number Theory - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Precalculus - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Geometry - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Number Theory - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## math_level1_Precalculus - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 123

# echo "## Experiment 3 - ${MODEL_ID} Seed 456"
# echo "###Mode without Chain of Thought"
# echo "## math_level1_Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Geometry - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Number Theory - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Precalculus - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456
# echo "## math_level1_Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Geometry - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Number Theory - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## math_level1_Precalculus - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "###Mode with Chain of Thought"
# echo "## math_level1_Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Counting & Probability - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Geometry - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Number Theory - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Prealgebra - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Precalculus - Original - Fewshot - GCP"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_gcp_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Counting & Probability - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Counting & Probability" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Geometry - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Geometry"\
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Intermediate Algebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Intermediate Algebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Number Theory - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Number Theory" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Prealgebra - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Prealgebra" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456

# echo "## math_level1_Precalculus - Original - Fewshot - AZR"
# python src/evaluate.py --model_name $MODEL_ID \
#                --dataset_name "math_level1_azr_Precalculus" \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --cot True \
#                --seed 456
