"script"
from dataclasses import dataclass, field
from typing import Optional, Dict, Union

@dataclass
class ModelConfig:
    """
    Configuration class for model settings.
    """
    model_name: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "The model that you want to train from the Hugging Face hub"}
    )
    dataset_name: str = field(
        default="vietgpt/wikipedia_vi",
        metadata={"help": "The instruction dataset to use"}
    )
    lang: str = field(
        default="vi",
        metadata={"help": "Language of the dataset to use (e.g. vi, ind, kr, ...)"}
    )
    dataset_dir: str = field(
        default="./datasets",
        metadata={"help": "The default directory for loading dataset"}
    )
    config_dir: str = field(
        default="./config",
        metadata={"help": "Configuration directory where contains LLM template,"
        "prompt template, generation configuration"}
    )
    output_dir: str = field(
        default="./results/generation",
        metadata={"help": "Output directory where the model"
        "predictions and checkpoints will be stored"}
    )
    output_eval_dir: str = field(
        default="./results/evaluation",
        metadata={"help": "The output folder to save metric scores"}
    )

@dataclass
class BitsAndBytesConfig:
    """
    Configuration class for bits and bytes parameters.
    """
    use_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4-bit precision base model loading"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for"
        "4-bit base models (double quantization)"}
    )
    cpu_offload_gb: int = field(
        default=0,
        metadata={"help": "Amount of memory to offload to CPU"}
    )

@dataclass
class InferenceConfig:
    """
    Configuration class for inference settings.
    """
    tokens: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "ms_hub_token": None,
            "hf_hub_token": None
        },
        metadata={"help": "Token configuration"}
    )
    settings: Dict[str, Union[bool, int]] = field(
        default_factory=lambda: {
            "smoke_test": False,
            "fewshot_prompting": False,
            "num_fs": 5,
            "seed": 42,
            "continue_infer": False
        },
        metadata={"help": "Inference settings"}
    )
    wrapper: Dict[str, str] = field(
        default_factory=lambda: {
            "wtype": "hf",
            "ptemplate": "llama-2"
        },
        metadata={"help": "Wrapper configuration"}
    )

def default_general_config() -> Dict[str, Union[int, str]]:
    """
    Returns a dictionary with default configuration values for general settings.
    """
    return {
        "per_device_eval_batch_size": 1,
        "dtype": "half",
        "device": "cuda:0",
        "n_bootstrap": 2,
        "p_bootstrap": 1.0,
        "bs": 128
    }

@dataclass
class ScriptArguments:
    """
    Configuration class for script arguments.
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    bits_and_bytes: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    general_config: Dict[str, Union[int, str, float]] = field(
        default_factory=default_general_config
    )

    @property
    def seed(self) -> int:
        "seed"
        return self.inference_config.settings['seed']
    @seed.setter
    def seed(self, value: int):
        "seed"
        self.inference_config.settings['seed'] = value

    # Add methods to access nested attributes if needed
    @property
    def dataset_name(self) -> str:
        "dataset"
        return self.model_config.dataset_name

    @property
    def lang(self) -> str:
        "lang"
        return self.model_config.lang

    # You can add similar properties for other nested attributes if needed
    @property
    def dataset_dir(self) -> str:
        "dataset"
        return self.model_config.dataset_dir
    @property
    def output_eval_dir(self) -> str:
        "output"
        return self.model_config.output_eval_dir
    @property
    def config_dir(self) -> str:
        "config"
        return self.model_config.config_dir
