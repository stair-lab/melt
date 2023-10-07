# Description: Run information retrieval experiments

## mMARCO - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mmarco \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## mMARCO - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mmarco_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## mMARCO - Fairness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mmarco_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &


## mRobust - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mrobust \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## mRobust - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mrobust_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## mRobust - Fairness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name mrobust_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &