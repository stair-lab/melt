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

## Dataset configuration
### Using local dataset (Optional)
- First, create a folder named "datasets" (if you use another folder instead "datasets", please add argument "**--dataset_dir <YOUR_NEW_DIR>**)
- Second, put all your necessary dataset in the folder, the package allows extention "csv, json, txt", the folder structure:
```
    villm-eval/datasets--<your_dataset_name>----<your_dataset_name>_train.<ext>
                                             |
                                             ----<your_dataset_name>_test.<ext>
```
- Finally, open the file "configs/<your_language>/datasets_info.json" to add your dataset
### Configuring dataset info
After having your dataset, access to directory **config** to create a folder which is named a language code. Please refer ***confg/***
```json
{
    "<your_dataset_name": {
      "hf_hub_url": "<HF_URL> (Optional)",
      "ms_hub_url": "<MS_HUB_URL> (Optional)",
      "file_name": "<Directory_to_your_dataset> (Optional)",
      "subset": "<YOUR_SUBSET_DATA> (Optional)",
      "task": "<TASK_NAME> (Required)",
      "train_split": "<Name_subset_for_train_split> (Optional, default: 'train')",
      "test_split": "<Name_subset_for_train_split> (Optional, default: 'test')",
      "label": ["<List_all_your_labels_on_classification_task>"]
      "prompting_strategy": 0, #Select prompt that you specify in ***prompt_template.json*** in a specific task (Optional, default: 0)
      "columns": {
          "type_id": "type_id",
          "passages": "passages",
          "context": "context",
          "query": "query",
          "answer": "answer",
          "options": "options",
          "source": "source",
          "target": "target"
        }
    }
}
```
The above json is a description how to specify a dataset to operate this package. For first 3 fields, please select one of them to specify the source of your data. For field "columns", each task has some necessary column. If the column names of your dataset are similar to the key field in column, we can skip these fields. Please strictly follow the format standard on your new dataset. The list of task is as follow.
#### Summarization
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "source": "source",
          "target": "target"
        }
    }
}
```
- Example dataset: [WikiLingua](https://huggingface.co/datasets/GEM/wiki_lingua)
#### Question Answering
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "context": "context",
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [MLQA](https://huggingface.co/datasets/facebook/mlqa)
#### Openended Knowledge
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [OpenEnded Knowledge](https://huggingface.co/datasets/ura-hcmut/Open-ended_knowledge)
#### Multiple Choice With Context
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "context": "context",
          "query": "query",
          "answer": "answer",
          "options": "options"
        }
    }
}
```
- Example dataset: [MTPC_Context](https://huggingface.co/datasets/ura-hcmut/MTPC_Context)
#### Sentiment Analysis
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [Sample](https://huggingface.co/datasets/ura-hcmut/sentiment_analysis)
#### Text Classification
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [Emotion classification](https://huggingface.co/datasets/ura-hcmut/text_classification)
#### Toxic Detection
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [Toxic detection](https://huggingface.co/datasets/ura-hcmut/toxic_detection)
#### Translation
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "source": "source",
          "target": "target"
        }
    }
}
```
- Example dataset: [OPUS100](https://huggingface.co/datasets/vietgpt/opus100_envi)
#### Information Retrieval
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "type_id": "type_id",
          "passages": "passages",
          "query": "query",
          "answer": "answer"
        }
    }
}
```
- Example dataset: [Information Retrieval](https://huggingface.co/datasets/ura-hcmut/Information_Retrieval)
#### Reasoning
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer",
        }
    }
}
```
- Example dataset: [Synthetic Natural Reasoning](https://huggingface.co/datasets/ura-hcmut/synthetic_reasoning_natural)
#### Math
```json
{
    "<your_dataset_name": {
      ...
      "columns": {
          "query": "query",
          "answer": "answer",
        }
    }
}
```
- Example dataset: [MATH](https://huggingface.co/datasets/ura-hcmut/MATH)

For more references, please check in 2 folders ***config/vi/dataset_info.json*** and ***config/ind/dataset_info.json***

## Prompt template configuration
Please refer the example in file ***config/{language_code}/prompt_template.json***. You need to define prompt in both PROMPT_TEMPLATE and CALIBRATION_PROMPT (if the required). The answer format and answer key in CALIBRATION_PROMPT, please leave them empty. For each task, you can add as many as possible all type of prompt you would like, and specific in the ***prompting_strategy*** in ***dataset_info.json***

## Other configurations
### Summac Model
- You add model maps which will be used for evaluating SummaC, please refer to ***config/summac_model.json***

### LLM template
- You can add a new chat template for your current LLM (**Note**: The template only work when you use wrapper types such as "tgi", "vllm", "hf"). Please refer to ***config/llm_template.json***

### Metric configuration
```json
{
    "NERModel": "<Name Entity Recognition Model On HF",
    "BERTScoreModel": {
        "model_type": "bert-base-multilingual-cased"
    },
    "SummaCModel": "<Select one of evaluating SummaC model that you specify in ***summac_model.json***>",
    "ToxicityEvaluationModel": "<Toxicity detection model in your langauge on HF>"
}
```
Please refer to ***config/{language_code}/metric_configuration.json***
## Controlling Text Generation and Bias Evaluation

This section explains how to adjust text generation parameters and access resources for evaluating bias in the language model.

### Customizing Generation Parameters

You can fine-tune the text generation process by adjusting specific parameters. The available parameters depend on the chosen wrapper type.  

**Important:** Pay close attention to the wrapper type you are using, as each supports a different set of parameters. For example, the `GeminiWrapper` does not support `top_k` but allows you to configure `top_p`. 

For a complete list of available parameters and their descriptions, please refer to the language-specific configuration file:

**`config/{language_code}/generation_config.json`**

### Evaluating Bias in Language Models

This framework provides resources for evaluating potential bias related to adjectives, professions, and gender. These resources consist of word lists organized by topic:

* **Adjectives:** A compilation of adjectives commonly used to describe individuals or groups.
* **Professions:** A list of various professions, occupations, and job titles.
* **Gender:**  Words related to different genders and gender identities. 

These word lists are located in the following directory: 

**`config/{language_code}/words`**

You can utilize these resources to assess and mitigate potential biases present in the language model's output. 



## Running pipeline
### Setup necessary environment variables
First, rename your ***env.template*** to ***.env***
Depend on wrapper type, it have other environment variables
- ***openai***
```
OPENAI_API_KEY="your OpenAI key"
```
- ***openai*** (in case of your provider from AzureGPT)
```
OPENAI_API_TYPE="azure"
OPENAI_API_BASE="https://<your-endpoint.openai.azure.com/>"
OPENAI_API_KEY="your AzureOpenAI key"
OPENAI_API_VERSION="2023-05-15"
```

- ***tgi***
```
TGI_ENDPOINT="http://localhost:10025"
```
- ***gemini***
```
GEMINI_KEY="random_api_Keyyyyyy"
```

### Run on local computer
**HF loading**
```bash
vieval --wtype hf \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --num_fs 3 \
               --fewshot_prompting True \
               --ptemplate mistral \
               --lang vi \
               --seed 42
```
**VLLM**
```bash
vieval --wtype vllm \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --num_fs 3 \
               --fewshot_prompting True \
               --ptemplate mistral \
               --lang vi \
               --seed 42
```
### Run on TGI
```bash
vieval --wtype tgi \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --fewshot_prompting True \
               --seed 42 \
               --ptemplate mistral \
               --lang vi \
```
### Run on GPT (gpt-3.5-turbo, gpt-4)
```bash
vieval --wtype openai \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```

### Run on Gemini
```bash
vieval --wtype gemini \
               --model_name gemini-pro \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```

### List of arguments
```bash
vieval [-h] [--model_name MODEL_NAME] [--dataset_name DATASET_NAME] [--use_4bit [USE_4BIT]] [--bnb_4bit_compute_dtype BNB_4BIT_COMPUTE_DTYPE]
              [--bnb_4bit_quant_type BNB_4BIT_QUANT_TYPE] [--use_nested_quant [USE_NESTED_QUANT]] [--lang LANG] [--dataset_dir DATASET_DIR] [--config_dir CONFIG_DIR]
              [--output_dir OUTPUT_DIR] [--output_eval_dir OUTPUT_EVAL_DIR] [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE] [--ms_hub_token MS_HUB_TOKEN]
              [--hf_hub_token HF_HUB_TOKEN] [--smoke_test [SMOKE_TEST]] [--fewshot_prompting [FEWSHOT_PROMPTING]] [--num_fs NUM_FS] [--seed SEED]
              [--continue_infer [CONTINUE_INFER]] [--wtype WTYPE] [--ptemplate PTEMPLATE] [--device DEVICE] [--n_bootstrap N_BOOTSTRAP] [--p_bootstrap P_BOOTSTRAP]
              [--bs BS]

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        The model that you want to train from the Hugging Face hub (default: meta-llama/Llama-2-7b-chat-hf)
  --dataset_name DATASET_NAME
                        The instruction dataset to use (default: vietgpt/wikipedia_vi)
  --use_4bit [USE_4BIT]
                        Activate 4-bit precision base model loading (default: False)
  --bnb_4bit_compute_dtype BNB_4BIT_COMPUTE_DTYPE
                        Compute dtype for 4-bit base models (default: bfloat16)
  --bnb_4bit_quant_type BNB_4BIT_QUANT_TYPE
                        Quantization type (fp4 or nf4) (default: nf4)
  --use_nested_quant [USE_NESTED_QUANT]
                        Activate nested quantization for 4-bit base models (double quantization) (default: False)
  --lang LANG           Language of the dataset to use (e.g. vi, ind, kr, ...) (default: vi)
  --dataset_dir DATASET_DIR
                        The default directory for loading dataset (default: ./datasets)
  --config_dir CONFIG_DIR
                        Configuration directory where contains LLM template, prompt template, generation configuration (default: ./config)
  --output_dir OUTPUT_DIR
                        Output directory where the model predictions and checkpoints will be stored (default: ./results/generation)
  --output_eval_dir OUTPUT_EVAL_DIR
                        The output folder to save metric scores (default: ./results/evaluation)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU for evaluation (default: 1)
  --ms_hub_token MS_HUB_TOKEN
                        Microsoft Hub token (default: None)
  --hf_hub_token HF_HUB_TOKEN
                        Hugging Face Hub token (default: None)
  --smoke_test [SMOKE_TEST]
                        Run a smoke test on a small dataset (default: False)
  --fewshot_prompting [FEWSHOT_PROMPTING]
                        Enable few-shot prompting (default: False)
  --num_fs NUM_FS       Number of samples for few-shot learning (default: 5)
  --seed SEED           Random seed (default: 42)
  --continue_infer [CONTINUE_INFER]
                        Wheather to continue previous inference process (default: False)
  --wtype WTYPE         Select type of wrapper: hf, tgi, azuregpt, gemini (default: hf)
  --ptemplate PTEMPLATE
                        Prompting template in chat template: llama-2, mistral, ... (default: llama-2)
  --device DEVICE       CUDA device (default: cuda:0)
  --n_bootstrap N_BOOTSTRAP
                        n bootstrap (default: 2)
  --p_bootstrap P_BOOTSTRAP
                        p bootstrap (default: 1.0)
  --bs BS               Bias metric (default: 128)
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
