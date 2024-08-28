# MELT: Multilingual Evaluation Toolkits

<div align="center">
<img src="melt_logo.jpg" alt="Project MELT" width="400"/>
</div>

The recent emergence of multilingual large language models (LLMs) is revolutionizing natural language processing, bridging communication gaps across diverse cultures and languages. However, to truly harness the potential of these models, it's crucial to understand their strengths and limitations across a wide range of languages and tasks. MELT is designed with this in mind, offering a comprehensive approach to evaluate LLMs in various linguistic contexts. Recognizing that proficiency in one language or task does not guarantee similar performance elsewhere, MELT enables users to pinpoint specific areas for improvement, fostering the development of robust and reliable multilingual language technologies.

MELT includes ten carefully selected evaluation scenarios, each targeting a key aspect of LLM capability:

1. **Summarization:** Evaluates the model's ability to condense large texts while retaining essential information.
2. **Question-Answering:** Assesses comprehension and accurate extraction of answers from provided contexts.
3. **Knowledge:** Tests the model's ability to recall and apply information across different domains.
4. **Sentiment Analysis:** Measures the ability to detect and classify emotional tones in text.
5. **Text Classification:** Evaluates accuracy in categorizing text into predefined labels.
6. **Toxic Detection:** Identifies the model's capacity to flag harmful or biased language.
7. **Language Modeling:** Tests fluency and coherence in generating contextually appropriate text.
8. **Reasoning:** Measures logical deduction and understanding of complex relationships.
9. **Math:** Assesses competency in solving mathematical problems in text form.
10. **Information Retrieval:** Tests effectiveness in searching, retrieving, and synthesizing relevant information.

MELT also includes tools to ensure the ethical deployment of LLMs:

- **Bias Assessment:** Identifies and mitigates potential biases in model outputs.
- **Toxicity Assessment:** Measures and controls the generation of harmful or offensive language.
- **Fairness Evaluation:** Ensures equitable performance across demographic groups and languages.
- **Robustness Analysis:** Examines resilience to noisy inputs and adversarial attacks, ensuring reliable performance in real-world scenarios.

MELT offers a holistic evaluation framework that not only assesses performance but also emphasizes ethical considerations, making it an essential tool for developing multilingual language models that are both effective and responsible. MELT currently supports the following languages and tasks:

| Task                      | Vietnamese | Indonesian   |  Korean  |
| :------------------------:| :---------:| :---------:  | :-----:  |
| Summarization             |    ✅      |      ✅     |   ✅     |
| Question Answering        |    ✅      |             |    ✅    |
| Sentiment Analysis        |    ✅      |      ✅     |          |
| Text Classification       |    ✅      |      ✅     |          |
| Toxicity Detection        |    ✅      |      ✅     |          |
| Open-ended Knowledge      |    ✅      |             |    ✅    |
| Multiple Choice Knowledge |    ✅      |             |           |
| Translation               |    ✅      |       ✅    |   ✅     |
| Reasoning                 |    ✅      |             |           |
| Math                      |    ✅      |             |           |
| Information Retrieval     |    ✅      |             |           |

MELT utilizes various metrics to ensure comprehensive evaluation:
- **[SummaC Model](https://arxiv.org/abs/2111.09525) (`config/summac_model.json`):** Add model maps for SummaC evaluation.
- **LLM Template (`config/llm_template.json`):** Define chat templates for specific LLM wrappers (e.g., `tgi`, `vllm`, `hf`).
- **Metric Configuration (`config/{language_code}/metric_configuration.json`):** Specify models for NER, [BERTScore](https://arxiv.org/abs/1904.09675), SummaC, and Toxicity evaluation.
- **Professions Words (`config/{language_code}/words/professions.txt`):**  List of vocabulary of occupation
- **Gender Words (`config/{language_code}/words/male.txt | female.txt`):** List of vocabulary of genders (male/female)
- **Adjective Words (`config/{language_code}/words/adjective.txt`):** List of adjective words
- **Token Pattern (`config/{language_code}/words/token_pattern.txt`):** List of possible token in a specific language. 

Explore MELT’s performance leaderboard at [​​https://ai.stanford.edu/~sttruong/villm/](https://ai.stanford.edu/~sttruong/villm/).

##Dataset Generation with MELT-chat

In addition to the aforementioned resources, we offer MELT-chat, an interactive user interface designed to facilitate dataset generation through direct interaction with large language models (LLMs). MELT-chat empowers users to engage in conversations with various LLMs and leverage their capabilities to produce tailored datasets. Users can interact with the language model in [here](https://www.ura.hcmut.edu.vn/melt/). For more details, please refer to [MELT-chat](https://github.com/stair-lab/fastchat). 

## Configuration
To get started, install the package:
```bash
   pip install -e .
```

To begin evaluating a new language, start by creating a folder for it within the `config` directory. Use language codes like "vi" for Vietnamese or "ind" for Indonesian as folder names. Once the folder is set up, proceed to configure your language dataset by following the instructions in the subsequent sections. This setup is essential to successfully run the evaluation process. 

One component of an evaluation pipeline is the instruction prompt. You need to define your prompt templates in `config/{language_code}/prompt_template.json`. You can define multiple prompts per task and select the desired one using the `"prompting_strategy"` field in `datasets_info.json`. Another component of the evaluation pipeline is the datasets. The `configs/<your_language>/dataset_info.json` file defines dataset configurations. Each dataset entry should follow this structure:
```json
{
    "<your_dataset_name>": {
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

- **Required Fields:**
    - `"task"`: Specifies the task type (e.g., "summarization", "question_answering"). See the supported tasks and their required columns in the following sections. 

- **Source Fields (Choose One):**
    - `"hf_hub_url"`: URL of the dataset on the Hugging Face Hub.
    - `"ms_hub_url"`: URL of the dataset on the Microsoft Hub.
    - `"file_name"`: Path to your local dataset file.

- **Optional Fields:**
    - `"subset"`:  Subset of the data to use (if applicable).
    - `"train_split"`/`"test_split"`: Custom names for train/test splits (defaults: "train"/"test").
    - `"prompting_strategy"`: Index of the prompt template to use from `prompt_template.json` (default: 0).
    - `"columns"`:  Mapping of column names in your dataset to the standard column names used by MELT Evaluation. If your dataset uses the standard names, you can omit this field.

Below, we outline the dataset format for our 10 core scenarios:
|                           | Task                      | Column                   | Dataset example                                              |
| :---------------:         | :------------------------:| :---------:              | :---------:                                                  | 
| Summarization             | `summarization`  |  `source`, `target`      | [WikiLingua](https://huggingface.co/datasets/GEM/wiki_lingua)| 
| Question Answering        | `question-answering` | `context`, `query`, `answer`| [MLQA](https://huggingface.co/datasets/facebook/mlqa) |
| Sentiment Analysis        | `sentiment-analysis` | `query`, `answer` |   [VSFC](https://huggingface.co/datasets/ura-hcmut/sentiment_analysis)         |  
| Text Classification       | `text-classification`| `query`, `answer` |   [VSMEC](https://huggingface.co/datasets/ura-hcmut/text_classification)  | 
| Toxicity Detection        | `toxic-detection`    | `query`, `answer` |   [ViHSD](https://huggingface.co/datasets/ura-hcmut/toxic_detection)|
| Open-ended Knowledge      | `knowledge-openended`| `query`, `answer` |   [zalo_e2eqa](https://huggingface.co/datasets/ura-hcmut/Open-ended_knowledge)  | 
| Multiple Choice Knowledge | `knowledge-mtpchoice`|  `context`, `query`, `answer`, `options` |   [ViMMRC](https://huggingface.co/datasets/ura-hcmut/MTPC_Context)         |
| Translation               | `translation`        |  `source`, `target`         | [OPUS100](https://huggingface.co/datasets/vietgpt/opus100_envi) |
| Reasoning                 | `reasoning`          |  `query`, `answer`         | [Synthetic Natural Reasoning](https://huggingface.co/datasets/ura-hcmut/synthetic_reasoning_natural) |
| Math                      |  `math`              |  `type_id`, `query`, `answer`    | [MATH](https://huggingface.co/datasets/ura-hcmut/MATH)|
| Information Retrieval     | `information-retrieval` | `type_id`, `passages`, `query`, `answer` | [Information Retrieval](https://huggingface.co/datasets/ura-hcmut/Information_Retrieval) |

We also support using local datasets via the following steps:
1. Create a "datasets" folder (or specify a custom directory using the `--dataset_dir` argument).
2. Place your datasets (`.csv`, `.json`, `.txt`) within the folder, structured as follows:
   ```
   melt/datasets--<your_dataset_name>----<your_dataset_name>_train.<ext>
                                             |
                                             ----<your_dataset_name>_test.<ext>
   ```
3. Add your dataset information to `configs/<your_language>/datasets_info.json`.

## Execution
First, one would need to configure the environment variables. Rename `.env.template` to `.env` and set up the required environment variables based on your chosen LLM wrapper:
- **OpenAI (`OPENAI_API_KEY`)**
- **AzureGPT (`OPENAI_API_TYPE`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_API_VERSION`)**
- **TGI (`TGI_ENDPOINT`)**
- **Gemini (`GEMINI_KEY`)** 

Then, we can use the `vieval` command with appropriate arguments to run the evaluation pipeline. 

**Example:**

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
**TGI**
```bash
vieval --wtype tgi \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --fewshot_prompting True \
               --seed 42 \
               --ptemplate mistral \
               --lang vi \
```
**GPT (gpt-3.5-turbo, gpt-4)**
```bash
vieval --wtype openai \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```

**Gemini**
```bash
vieval --wtype gemini \
               --model_name gemini-pro \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```
**List of arguments**
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
## Q&A
Our software is designed to offer comprehensive functionality for evaluating large language models (LLMs). It includes a range of metrics and analyses to assess LLM performance across different languages. The software supports flexible datasets, allowing users to load data from various formats, either locally or from the Hugging Face Hub. Additionally, it provides customizable prompting options, enabling users to define and select multiple prompt templates tailored to specific tasks. To accommodate different user preferences, the software is compatible with various LLM wrappers, including Hugging Face Transformers, Text Generation Inference, OpenAI, and Gemini.

However, it's important to note that some functionalities fall outside the scope of our software. Specifically, it does not support LLM training; the focus is solely on evaluation. Furthermore, while the tool can provide valuable insights for model fine-tuning, it does not offer automatic hyperparameter optimization for LLMs.

If you encounter any issues, please submit them via our GitHub page at [https://github.com/stair-lab/melt/issues](https://github.com/stair-lab/melt/issues). This project is licensed under the MIT License—details can be found in the [LICENSE](./LICENSE) file. We follow the MELT Code of Conduct, outlined in the [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) file. If you find this project helpful, please consider citing us in your work.

```
@inproceedings{crossing2024,
    title = "Crossing Linguistic Horizons: Finetuning and Comprehensive Evaluation of Vietnamese Large Language Models",
    author = "Truong, Sang T.  and Nguyen, Duc Q.  and Nguyen, Toan D. V.  and Le, Dong D.  and Truong, Nhi N.  and Quan, Tho  and Koyejo, Sanmi",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = June,
    year = "2024",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
}
```
