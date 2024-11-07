from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ScriptArguments:
    """
    ScriptArguments:
    Contains all configuration parameters for the text generation pipeline, including model,
    dataset, training options, and inference settings.
    """

    # Model and Dataset
    model_name: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "The model that you want to train from the Hugging Face hub"},
    )
    tasks: str = field(
        default="vietgpt/wikipedia_vi",
        metadata={"help": "The instruction dataset to use"},
    )

    # Bitsandbytes Parameters
    use_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Activate 4-bit precision base model loading"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Compute dtype for 4-bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False, metadata={"help": "Activate nested quantization for 4-bit base models"},
    )
    cpu_offload_gb: int = field(
        default=0, metadata={"help": "Amount of memory to offload to CPU"},
    )

    # Dataset and Language
    lang: str = field(
        default="vi", metadata={"help": "Language of the dataset to use (e.g., vi, ind, kr, ...)"},
    )
    dataset_dir: str = field(
        default="./datasets", metadata={"help": "The default directory for loading dataset"},
    )
    config_dir: str = field(
        default="./config", metadata={"help": "Configuration directory for templates and configs"},
    )

    # Output and Evaluation
    output_dir: str = field(
        default="./results/generation", metadata={"help": "Directory for model predictions and checkpoints"},
    )
    output_eval_dir: str = field(
        default="./results/evaluation", metadata={"help": "The output folder to save metric scores"},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size per GPU for evaluation"},
    )
    dtype: str = field(
        default="half", metadata={"help": "Data type for model loading"},
    )

    # Inference Parameters
    ms_hub_token: Optional[str] = field(
        default=None, metadata={"help": "Microsoft Hub token"},
    )
    hf_hub_token: Optional[str] = field(
        default=None, metadata={"help": "Hugging Face Hub token"},
    )
    smoke_test: Optional[bool] = field(
        default=False, metadata={"help": "Run a smoke test on a small dataset"},
    )
    prompt_type: Literal["normal", "weaker", "medium"] = field(
        default="normal", metadata={"help": "Prompting type: normal, weaker, medium"},
    )
    category: Literal[
        "zero-shot", "few-shot", "robustness-aware", "fairness-aware", "chain-of-thought", "randomized-choice"
    ] = field(
        default="zero-shot", metadata={"help": "Prompting category"},
    )
    num_fewshot: Optional[int] = field(
        default=2, metadata={"help": "Number of samples for few-shot learning"},
    )
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})

    # Additional Options
    continue_infer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to continue the previous inference process"},
    )
    model: str = field(
        default="hf", metadata={"help": "Select type of wrapper: hf, tgi, azuregpt, gemini"},
    )
    ptemplate: Optional[str] = field(
        default="llama-2", metadata={"help": "Prompting template in chat template"},
    )
    device: str = field(default="cuda:0", metadata={"help": "CUDA device"})
    n_bootstrap: int = field(default=2, metadata={"help": "n bootstrap"})
    p_bootstrap: float = field(default=1.0, metadata={"help": "p bootstrap"})
    bs: int = field(default=128, metadata={"help": "Bias metric"})
