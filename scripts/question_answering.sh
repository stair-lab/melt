# Description: Run question-answering experiments

## xquad_xtreme - Original
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme \
               --prompting_strategy 2 &

## xquad_xtreme - Robustness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_robustness \
               --prompting_strategy 2 &

## xquad_xtreme - Fairness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name xquad_xtreme_fairness \
               --prompting_strategy 2 &


## MLQA - Original
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/mlqa \
               --prompting_strategy 2 &

## MLQA - Robustness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_robustness \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_robustness \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/mlqa_robustness \
               --prompting_strategy 2 &

## MLQA - Fairness
python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_fairness \
               --prompting_strategy 0 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name mlqa_fairness \
               --prompting_strategy 1 &

python test.py --model_name martinakaduc/llama-2-7b-hf-vi \
               --dataset_name GEM/mlqa_fairness \
               --prompting_strategy 2 &