"""
This module defines the `ScriptArguments` class used for configuring script parameters.

The `ScriptArguments` class utilizes Python's `dataclass` to provide a 
structured way to handle various configuration settings 
needed for running the script. The fields within this 
class include parameters for model and dataset configuration, 
precision and quantization settings, output directories, and inference parameters.

Class:
    ScriptArguments: A data class that encapsulates various 
    configuration parameters for the script.


Attributes:
    model_name (str): The model name to train or use, typically from the Hugging Face hub.
    dataset_name (str): The dataset name to use for training or evaluation.
    use_4bit (Optional[bool]): Whether to use 4-bit precision for model loading.
    bnb_4bit_compute_dtype (Optional[str]): Data type for 4-bit model computation.
    bnb_4bit_quant_type (Optional[str]): Quantization type (e.g., fp4 or nf4).
    use_nested_quant (Optional[bool]): Whether to use nested quantization.
    cpu_offload_gb (int): Amount of memory to offload to CPU.
    lang (str): Language of the dataset (e.g., vi, ind, kr).
    dataset_dir (str): Directory for loading datasets.
    config_dir (str): Directory for configuration files.
    output_dir (str): Directory for saving model predictions and checkpoints.
    output_eval_dir (str): Directory for saving evaluation metrics.
    per_device_eval_batch_size (Optional[int]): Batch size per GPU for evaluation.
    dtype (str): Data type for model loading.
    ms_hub_token (Optional[str]): Token for Microsoft Hub.
    hf_hub_token (Optional[str]): Token for Hugging Face Hub.
    smoke_test (Optional[bool]): Whether to run a smoke test on a small dataset.
    fewshot_prompting (Optional[bool]): Whether to enable few-shot prompting.
    num_fs (Optional[int]): Number of samples for few-shot learning.
    seed (Optional[int]): Random seed for reproducibility.
    continue_infer (Optional[bool]): Whether to continue a previous inference process.
    wtype (str): Type of wrapper to use (e.g., hf, tgi, azuregpt, gemini).
    ptemplate (Optional[str]): Prompting template to use (e.g., llama-2, mistral).
    device (str): CUDA device to use.
    n_bootstrap (int): Number of bootstrap samples.
    p_bootstrap (float): Probability for bootstrap sampling.
    bs (int): Bias metric.

This class serves as a configuration container to manage and pass 
parameters throughout the script efficiently.
"""

from dataclasses import dataclass, field
from typing import Optional
from typing import Dict

@dataclass
class ModelConfig:
    """
    Configuration class for model settings.

    Attributes:
        model_name (str): The name of the model to train from the Hugging Face hub.
        dataset_name (str): The instruction dataset to use.
        lang (str): Language of the dataset (e.g., vi, ind, kr, ...).
        dataset_dir (str): Default directory for loading datasets.
        config_dir (str): Directory containing LLM template, 
        prompt template, and generation configuration.
        output_dir (str): Directory for storing model predictions and checkpoints.
        output_eval_dir (str): Directory for saving metric scores.
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

    This class contains settings related to the precision and quantization of 
    base models, including activation of 4-bit precision, compute data type,
    quantization type, nested quantization, and CPU offloading settings.

    Attributes:
        use_4bit (Optional[bool]): Whether to activate 4-bit precision base model loading.
        bnb_4bit_compute_dtype (Optional[str]): Compute data 
        type for 4-bit base models (e.g., 'bfloat16').
        bnb_4bit_quant_type (Optional[str]): Quantization type 
        used for 4-bit models (e.g., 'fp4' or 'nf4').
        use_nested_quant (Optional[bool]): Whether to activate 
        nested quantization for 4-bit base models.
        cpu_offload_gb (int): Amount of memory to offload to CPU, in gigabytes.
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

    Attributes:
        tokens (Dict[str, Optional[str]]): Configuration for tokens 
        including Microsoft Hub and Hugging Face Hub tokens.
        settings (Dict[str, Optional]): Inference settings including 
        smoke test, few-shot prompting, number of few-shot samples, 
        random seed, and whether to continue previous inference.
        wrapper (Dict[str, str]): Wrapper configuration 
        including the type of wrapper and prompting template.
    """
    tokens: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "ms_hub_token": None,
            "hf_hub_token": None
        },
        metadata={"help": "Token configuration"}
    )
    settings: Dict[str, Optional] = field(
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

def default_general_config():
    """
    Returns a dictionary with default configuration values for general settings.

    This function provides default values for various configuration parameters
    related to general settings, such as batch size, data type, device, and
    other metrics.

    Returns:
        dict: A dictionary containing default values for:
            - per_device_eval_batch_size: The batch size per GPU for evaluation.
            - dtype: The data type for model loading.
            - device: The CUDA device to be used.
            - n_bootstrap: The number of bootstrap iterations.
            - p_bootstrap: The probability for bootstrap sampling.
            - bs: Bias metric.
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

    Attributes:
        model_config (ModelConfig): Configuration for model settings.
        bits_and_bytes (BitsAndBytesConfig): Configuration for bits and bytes parameters.
        inference_config (InferenceConfig): Configuration for inference settings.
        general_config (Dict[str, Optional]): General configuration settings including
            batch size, data type, device, and other metrics.
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    bits_and_bytes: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    general_config: Dict[str, Optional] = field(default_factory=default_general_config)
