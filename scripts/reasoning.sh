# Description: Run reasoning experiments

## synthetic_natural - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_natural_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_induction - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_induction_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_pattern_match - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_pattern_match_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_variable_substitution - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_variable_substitution_gcp \
               --prompting_strategy 0 \
               --fewshot_prompting True &





## synthetic_natural - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_natural_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_induction - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_induction_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_pattern_match - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_pattern_match_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## synthetic_variable_substitution - Original - Fewshot
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name synthetic_variable_substitution_azr \
               --prompting_strategy 0 \
               --fewshot_prompting True &





## math_level1_Algebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Counting & Probability - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Geometry - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Intermediate Algebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Number Theory - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Prealgebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Precalculus - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True &






## math_level1_Algebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Counting & Probability - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Geometry - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Intermediate Algebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Number Theory - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Prealgebra - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Precalculus - Original - Fewshot - GCP
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_gcp_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &




## math_level1_Algebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Counting & Probability - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Geometry - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Intermediate Algebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Number Theory - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Prealgebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## math_level1_Precalculus - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True &





## math_level1_Algebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Counting & Probability - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Geometry - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Intermediate Algebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Number Theory - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Prealgebra - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &

## math_level1_Precalculus - Original - Fewshot - AZR
python test.py --model_name ura-hcmut/ura-llama-7b-r128 \
               --dataset_name "math_level1_azr_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --cot True &