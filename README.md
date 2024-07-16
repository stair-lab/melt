# ViLLM Evaluation

## Overview

## Installation
Initialize environment:
```bash
conda create -n villm python=3.10
conda activate villm
```
**Install PyTorch (with CUDA 12.1):**

**Recommended:** Visit the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for the most up-to-date instructions.

**Alternative (if you have CUDA 12.1 set up):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Install package:
```bash
pip install -e .
```

## Running pipeline
### Run on local computer
```bash
vieval --wtype hf \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --ptemplate mistral \
               --seed 42
```
### Run on TGI
```bash
vieval --wtype tgi \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --ptemplate mistral \
               --tgi http://127.0.0.1:10025
```
### Run on GPT (gpt-3.5-turbo, gpt-4)
```bash
vieval --wtype azuregpt \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42
```

### Run on Gemini
```bash
vieval --wtype gemini \
               --model_name gemini-pro \
               --dataset_name zalo_e2eqa \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42
```
## Citation
```
@inproceedings{crossing2024,
    title = "Crossing Linguistic Horizons: Finetuning and Comprehensive Evaluation of Vietnamese Large Language Models",
    author = "Truong, Sang T.  and Nguyen, Duc Q.  and Nguyen, Toan D. V.  and Le, Dong D.  and Truong, Nhi N.  and Quan, Tho  and Koyejo, Sanmi",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = June,
    year = "2024",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "",
    pages = "",
}
```
