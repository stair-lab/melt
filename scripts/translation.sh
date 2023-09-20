# Description: Run translation experiments

## PhoMT - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoMT_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoMT_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True

## PhoMT - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoMT_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name PhoMT_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## OPUS100 - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name opus100_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name opus100_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True

## OPUS100 - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name opus100_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name opus100_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True