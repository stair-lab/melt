# Welcome to the MELT Evaluation Project!

MELT Evaluation is a comprehensive package designed for evaluating Large Language Models (LLMs) in a specific language. By providing insightful metrics and analyses, our tool empowers you to:

- **Fine-tune your LLMs:** Leverage the evaluation results to fine-tune your LLMs for optimal performance using tools like [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory).
- **Deploy with confidence:** Easily deploy your fine-tuned LLMs for real-world applications using [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

MELT is hosted by [Stanford AI Lab](https://ai.stanford.edu/) 

## Getting Started

### Installation

1. **Initialize environment:**
   ```bash
   conda create -n melt python=3.10
   conda activate melt
   ```

2. **Install PyTorch (with CUDA 12.1):**
   - **Recommended:** Visit [https://pytorch.org/](https://pytorch.org/) for the latest instructions.
   - **Alternative (CUDA 12.1 already set up):**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

3. **Install MELT Evaluation:**
   ```bash
   pip install -e .
   ```

### Dataset Configuration

#### Using a Local Dataset (Optional)

1. Create a "datasets" folder (or specify a custom directory using the `--dataset_dir` argument).
2. Place your datasets (`.csv`, `.json`, `.txt`) within the folder, structured as follows:
   ```
   melt/datasets--<your_dataset_name>----<your_dataset_name>_train.<ext>
                                             |
                                             ----<your_dataset_name>_test.<ext>
   ```
3. Add your dataset information to `configs/<your_language>/datasets_info.json` (see **Configuring Dataset Info** below).

#### Configuring Dataset Info

The `configs/<your_language>/datasets_info.json` file defines dataset configurations. Each dataset entry should follow this structure:

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

#### Supported Tasks and Data Formats

##### Summarization (task: "summarization") 

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "summarization", 
      "columns": {
          "source": "<column_name_for_source_text>",
          "target": "<column_name_for_summary_text>"
        }
    }
}
```

- **Example Dataset:** [WikiLingua](https://huggingface.co/datasets/GEM/wiki_lingua)

**Data Format:**

```
{"source": "Alice went to the store to buy some milk.", "target": "Alice went shopping for milk."}
{"source": "Bob is a programmer. He works at Google.", "target": "Bob, a programmer, is employed by Google."}
// ... more examples
```

##### Question Answering (task: "question-answering")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "question-answering", 
      "columns": {
          "context": "<column_name_for_context_text>",
          "query": "<column_name_for_question>",
          "answer": "<column_name_for_answer_text>"
        }
    }
}
```

- **Example Dataset:** [MLQA](https://huggingface.co/datasets/facebook/mlqa)

**Data Format:**

```
{"context": "The cat sat on the mat. The mat is blue.", "query": "Where is the cat?", "answer": {"text": ["on the mat"]}}
{"context": "The Eiffel Tower is in Paris.", "query": "What city is the Eiffel Tower in?", "answer": {"text": ["Paris"]}}
// ... more examples
```

##### Open-ended Knowledge (task: "knowledge-openended")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "knowledge-openended",
      "columns": {
          "query": "<column_name_for_question>",
          "answer": "<column_name_for_answer_text>"
        }
    }
}
```

- **Example Dataset:** [OpenEnded Knowledge](https://huggingface.co/datasets/ura-hcmut/Open-ended_knowledge)

**Data Format:**

```
{"query": "What is the capital of France?", "answer": "Paris"}
{"query": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"} 
// ... more examples
```

##### Multiple Choice with Context (task: "knowledge-mtpchoice")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "knowledge-mtpchoice", 
      "columns": {
          "context": "<column_name_for_context_text>",
          "query": "<column_name_for_question>",
          "answer": "<column_name_for_correct_answer_option>",
          "options": "<column_name_for_list_of_answer_options>" 
        }
    }
}
```

- **Example Dataset:** [MTPC_Context](https://huggingface.co/datasets/ura-hcmut/MTPC_Context)

**Data Format:**

```
{"context": "The sky is blue because of Rayleigh scattering.", "query": "Why is the sky blue?", "answer": "C", "options": ["Reflection", "Refraction", "Rayleigh scattering", "Diffraction"]}
// ... more examples
```

##### Sentiment Analysis (task: "sentiment-analysis")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "sentiment-analysis",
      "columns": {
          "query": "<column_name_for_text>",
          "answer": "<column_name_for_sentiment_label>" 
        }
    }
}
```

- **Example Dataset:** [Sample](https://huggingface.co/datasets/ura-hcmut/sentiment_analysis)

**Data Format:**

```
{"query": "This movie was amazing!", "answer": "1"}
{"query": "I'm feeling really down today.", "answer": "0"}
// ... more examples
```

##### Text Classification (task: "text-classification")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "text-classification",
      "columns": {
          "query": "<column_name_for_text>",
          "answer": "<column_name_for_class_label>" 
        }
    }
}
```

- **Example Dataset:** [Emotion classification](https://huggingface.co/datasets/ura-hcmut/text_classification)

**Data Format:**

```
{"query": "This is the best day ever!", "answer": "0"}
{"query": "I'm so scared right now.", "answer": "1"}
// ... more examples 
```

##### Toxic Detection (task: "toxic-detection")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "toxic-detection", 
      "columns": {
          "query": "<column_name_for_text>",
          "answer": "<column_name_for_toxicity_label>" 
        }
    }
}
```

- **Example Dataset:** [Toxic detection](https://huggingface.co/datasets/ura-hcmut/toxic_detection)

**Data Format:**

```
{"query": "You are an idiot!", "answer": "1"}
{"query": "Have a great day!", "answer": "0"}
// ... more examples
```

##### Translation (task: "translation")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "translation", 
      "columns": {
          "source": "<column_name_for_source_language_text>",
          "target": "<column_name_for_target_language_text>"
        }
    }
}
```

- **Example Dataset:** [OPUS100](https://huggingface.co/datasets/vietgpt/opus100_envi)

**Data Format:**

```
{"source": "Hello world", "target": "Xin chào thế giới"}
{"source": "How are you?", "target": "Bạn có khỏe không?"} 
// ... more examples
```

##### Information Retrieval (task: "information-retrieval")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "information-retrieval", 
      "columns": {
          "type_id": "<column_name_for_passage_type>", 
          "passages": "<column_name_for_list_of_passages>",
          "query": "<column_name_for_query>",
          "answer": "<column_name_for_relevant_passage_index>" 
        }
    }
}
```

- **Example Dataset:** [Information Retrieval](https://huggingface.co/datasets/ura-hcmut/Information_Retrieval)

**Data Format:**
- Training set
```
{ 
    "query": "User's search query", 
    "positive": "A relevant passage", 
    "negative": "An irrelevant passage"
}
// ... more examples
```
- Testing set
```
{
    "type_id": 1, 
    "query": "User's search query",
    "answer": [1],
    "passages": { 
        "id": [1, 2, 3], 
        "passage": ["Passage 1 text", "Passage 2 text", "Passage 3 text"] 
    }
    
// ... more examples
```
##### Reasoning (task: "reasoning")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "reasoning", 
      "columns": {
          "query": "<column_name_for_reasoning_problem_or_question>", 
          "answer": "<column_name_for_answer_or_solution>" 
        }
    }
}
```

- **Example Dataset:** [Synthetic Natural Reasoning](https://huggingface.co/datasets/ura-hcmut/synthetic_reasoning_natural)

**Data Format:**

```
{"query": "If A is taller than B, and B is taller than C, who is the shortest?", "answer": "C"}
// ... more examples 
```

##### Math (task: "math")

```json
{
    "<your_dataset_name>": {
      // ... other fields
      "task": "math", 
      "columns": {
          "query": "<column_name_for_math_problem>",
          "answer": "<column_name_for_answer>"
        }
    }
}
```

- **Example Dataset:** [MATH](https://huggingface.co/datasets/ura-hcmut/MATH)

**Data Format:**

```
{
    "query": "Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?", 
    "answer": "First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\boxed{238}$"}
{
    "query": "If $g(x) = x^2$ and $f(x) = 2x - 1$, what is the value of $f(g(2))$?", 
    "answer": "\[ f(g(2))=f\left(2^2\right)=f(4)=2\cdot4-1=\boxed{7} \]"
}
// ... more examples
```

#### Prompt Template Configuration

Define your prompt templates in `config/{language_code}/prompt_template.json`. You can define multiple prompts per task and select the desired one using the `"prompting_strategy"` field in `datasets_info.json`.

#### Other Configurations

- **SummaC Model (`config/summac_model.json`):** Add model maps for SummaC evaluation.
- **LLM Template (`config/llm_template.json`):** Define chat templates for specific LLM wrappers (e.g., "tgi", "vllm", "hf").
- **Metric Configuration (`config/{language_code}/metric_configuration.json`):** Specify models for NER, BERTScore, SummaC, and Toxicity evaluation.

#### Available datasets

| Task                      | Vietnamese | Indonesian | Korean |
| :------------------------:| :---------:| :---------:| :-----:|
| Summarization             |    ✅      |      ✅     |   ✅    |
| Question Answering        |    ✅      |             |    ✅   |`
| Sentiment Analysis        |    ✅      |      ✅     |        |
| Text Classification       |    ✅      |      ✅     |        |
| Toxicity Detection        |    ✅      |      ✅     |        |
| Open-ended Knowledge      |    ✅      |            |    ✅   |`
| Multiple Choice Knowledge |    ✅      |            |        |
| Translation               |    ✅      |       ✅    |   ✅    |
| Reasoning                 |    ✅      |            |        |
| Math                      |    ✅      |            |        |


## Running the Evaluation Pipeline

### Environment Variables

Rename `.env.template` to `.env` and set up the required environment variables based on your chosen LLM wrapper:

- **OpenAI (`OPENAI_API_KEY`)**
- **AzureGPT (`OPENAI_API_TYPE`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_API_VERSION`)**
- **TGI (`TGI_ENDPOINT`)**
- **Gemini (`GEMINI_KEY`)** 

### Running the Evaluation

Use the `melt` command with appropriate arguments to run the evaluation pipeline. 

**Example:**

```bash
melt --wtype hf \
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
melt --wtype vllm \
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
melt --wtype tgi \
               --model_name ura-hcmut/MixSUra \
               --dataset_name zalo_e2eqa \
               --fewshot_prompting True \
               --seed 42 \
               --ptemplate mistral \
               --lang vi \
```
**GPT (gpt-3.5-turbo, gpt-4)**
```bash
melt --wtype openai \
               --model_name gpt-4 \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```

**Gemini**
```bash
melt --wtype gemini \
               --model_name gemini-pro \
               --dataset_name zalo_e2eqa \
               --lang vi \
               --fewshot_prompting True \
               --seed 42
```
**List of arguments**
```bash
melt [-h] [--model_name MODEL_NAME] [--dataset_name DATASET_NAME] [--use_4bit [USE_4BIT]] [--bnb_4bit_compute_dtype BNB_4BIT_COMPUTE_DTYPE]
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
## Scope

### In Scope

- **Comprehensive LLM Evaluation:** Provides a suite of metrics and analyses for evaluating LLM performance in a specific language.
- **Dataset Flexibility:** Supports various dataset formats and loading from local files or Hugging Face Hub.
- **Customizable Prompting:** Allows defining and selecting from multiple prompt templates for different tasks.
- **Multiple LLM Wrappers:** Supports running evaluations with different LLM wrappers (Hugging Face Transformers, Text Generation Inference, OpenAI, Gemini).

### Out of Scope

- **LLM Training:** This tool focuses on evaluation and does not provide LLM training capabilities. 
- **Automatic Hyperparameter Optimization:** While the evaluation results can guide fine-tuning, this tool does not offer automatic hyperparameter optimization for LLMs.

## Contributing

We welcome contributions from the community! Please refer to our [Contributor Guide](./CONTRIBUTING.md) to get started.

## Communications

- **GitHub Issues:** [https://github.com/stair-lab/melt/issues](https://github.com/stair-lab/melt/issues)

**We are actively working on establishing more communication channels.**

## Resources

- **Leaderboard:** [https://ai.stanford.edu/~sttruong/MELT](https://ai.stanford.edu/~sttruong/melt)

## License

This project is licensed under MIT - see the [LICENSE](./LICENSE) file for details.

## Conduct

We adhere to the MELT Code of Conduct - see the [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) file for details. 

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