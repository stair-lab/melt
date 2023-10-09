# ./run_sentiment.sh 2>&1 | tee logs/log_sentiment.txt
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-13b

# Description: Run toxicity analysis experiments
echo "#Toxicity Experiment\n"
echo "## Experiment 1 - ${MODEL_ID} Seed 42"
echo "## ViCTSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## ViCTSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## ViCTSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## ViCTSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42

echo "## ViHSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42


echo "## ViHSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## ViHSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## ViHSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42

echo "## Experiment 2 - ${MODEL_ID} Seed 123"
echo "## ViCTSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## ViCTSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## ViCTSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## ViCTSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123

echo "## ViHSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 123


echo "## ViHSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## ViHSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## ViHSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 123
               
echo "## Experiment 3 - ${MODEL_ID} Seed 456"
echo "## ViCTSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## ViCTSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## ViCTSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## ViCTSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViCTSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456

echo "## ViHSD - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 456


echo "## ViHSD - Fairness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## ViHSD - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## ViHSD - Original - Zeroshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name ViHSD \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 456
