# Description: Run language modelling experiments

## MLQA_MLM - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_MLM \
               --prompting_strategy 0 \
               --fewshot_prompting True

## MLQA_MLM - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_MLM_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## MLQA_MLM - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_MLM_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True


## VSEC - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name VSEC \
               --prompting_strategy 0 \
               --fewshot_prompting True

## VSEC - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name VSEC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## VSEC - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name VSEC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True