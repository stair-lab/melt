# Description: Run knowledge experiments

## ZaloQA - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True

## ZaloQA - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name zalo_e2eqa_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## ZaloQA - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name zalo_e2eqa_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True


## ViMMRC - Original - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name ViMMRC \
               --prompting_strategy 0 \
               --fewshot_prompting True

## ViMMRC - Robustness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name ViMMRC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True

## ViMMRC - Fairness - Fewshot
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name ViMMRC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True