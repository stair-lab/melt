# Description: Run text classification experiments

## PhoATIS - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoATIS \
               --prompting_strategy 0 \
               --fewshot_prompting True

## PhoATIS - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoATIS_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## PhoATIS - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoATIS_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## UIT-VSMEC - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name UIT-VSMEC \
               --prompting_strategy 0 \
               --fewshot_prompting True

## UIT-VSMEC - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name UIT-VSMEC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## UIT-VSMEC - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name UIT-VSMEC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True