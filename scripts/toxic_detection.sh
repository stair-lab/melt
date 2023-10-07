# Description: Run toxic detection experiments

## ViCTSD - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## ViCTSD - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViCTSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## ViCTSD - Fairness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViCTSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &


## ViHSD - Original - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## ViHSD - Robustness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViHSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True &

## ViHSD - Fairness - Fewshot
python test.py --model_name $MODEL_ID \
               --dataset_name ViHSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True &