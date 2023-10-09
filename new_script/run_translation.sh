# ./run_sentiment.sh 2>&1 | tee logs/log_sentiment.txt
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="2,3"
export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=ura-hcmut/ura-llama-7b

# Description: Run translation experiments
#echo "#Translation Experiment\n"
#echo "## Experiment 1 - 7b-ura-llama Seed 42"
#echo "## PhoMT_envi - Original - Fewshot"
#python test.py --model_name ${MODEL_ID} \
#               --dataset_name PhoMT_envi \
#               --prompting_strategy 0 \
#               --fewshot_prompting True \
#               --seed 42

echo "## PhoMT_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42


echo "## PhoMT_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42

echo "## PhoMT_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42


echo "## opus100_envi - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42


echo "## opus100_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42


echo "## opus100_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42


echo "## opus100_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42


echo "## Experiment 2 - 7b-ura-llama Seed 123"
echo "## PhoMT_envi - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## PhoMT_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## PhoMT_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## PhoMT_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123


echo "## opus100_envi - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## opus100_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## opus100_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 123


echo "## opus100_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 123

echo "## Experiment 3 - 7b-ura-llama Seed 456"
echo "## PhoMT_envi - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## PhoMT_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## PhoMT_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## PhoMT_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name PhoMT_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## opus100_envi - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## opus100_envi - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_envi_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456

echo "## opus100_vien - Original - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 456


echo "## opus100_vien - Robustness - Fewshot"
python test.py --model_name ${MODEL_ID} \
               --dataset_name opus100_vien_robustness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 456
