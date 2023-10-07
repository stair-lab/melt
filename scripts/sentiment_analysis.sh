# Description: Run sentiment analysis experiments

## UIT-VSFC - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name UIT-VSFC \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## UIT-VSFC - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name UIT-VSFC_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## UIT-VSFC - Fairness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name UIT-VSFC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## vlsp2016 - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name vlsp2016 \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## vlsp2016 - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name vlsp2016_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## vlsp2016 - Fairness - Fewshot
python test.py --model_name $MODEL_ID
               --dataset_name vlsp2016_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &