# URA-LLaMa

## Overview

## Running pipeline
### Run on local computer
```bash
python src/evaluate.py --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42
```
### Run on TGI
```bash
python src/evaluate.py --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42
               --tgi http://127.0.0.1:10025
```
### Run on GPT (gpt-3.5-turbo, gpt-4)
```bash
python src/evaluate.py --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42
```

## License and Usage Agreement

## Citation
