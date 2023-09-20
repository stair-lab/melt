# Description: Run summarization experiments

## Vietnews - Original
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews \
               --prompting_strategy 2

## Vietnew - Robustness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_robustness \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_robustness \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_robustness \
               --prompting_strategy 2

## Vietnew - Fairness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_fairness \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_fairness \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name vietnews_fairness \
               --prompting_strategy 2


## WikiLingua - Original
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/wiki_lingua \
               --prompting_strategy 2

## WikiLingua - Robustness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua_robustness \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua_robustness \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/wiki_lingua_robustness \
               --prompting_strategy 2

## WikiLingua - Fairness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua_fairness \
               --prompting_strategy 0

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name wiki_lingua_fairness \
               --prompting_strategy 1

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/wiki_lingua_fairness \
               --prompting_strategy 2