# Description: Run reasoning experiments

## synthetic_natural - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name synthetic_natural \
               --prompting_strategy 0 \
               --fewshot_prompting True

## synthetic_induction - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name synthetic_induction \
               --prompting_strategy 0 \
               --fewshot_prompting True

## synthetic_pattern_match - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name synthetic_pattern_match \
               --prompting_strategy 0 \
               --fewshot_prompting True

## synthetic_variable_substitution - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name synthetic_variable_substitution \
               --prompting_strategy 0 \
               --fewshot_prompting True


## math_level1_Algebra - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Counting & Probability - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Counting & Probability" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Geometry - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Geometry"\
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Intermediate Algebra - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Intermediate Algebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Number Theory - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Number Theory - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Number Theory" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Prealgebra - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Prealgebra" \
               --prompting_strategy 0 \
               --fewshot_prompting True

## math_level1_Precalculus - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name "math_level1_Precalculus" \
               --prompting_strategy 0 \
               --fewshot_prompting True