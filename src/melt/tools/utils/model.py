"""
This module contains functions for loading pre-trained 
models and tokenizers with configuration options.

Functions:
- get_model(config): Loads a pre-trained causal language 
model and tokenizer based on the provided configuration.
    - config (object): An instance of a configuration 
    class that contains model parameters and settings.

Dependencies:
- torch: For handling model computation and GPU compatibility checks.
- transformers: For loading and configuring models and tokenizers.

Usage:
- Initialize a configuration object with necessary parameters.
- Call `get_model` with the configuration object to load the desired model and tokenizer.
"""
import torch

try:
    from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
except ImportError as e:
    print("Error importing 'transformers':", e)
else:
    print("Successfully imported 'transformers'.")

def get_model(config):
    """
    Loads a pre-trained causal language model and its associated tokenizer 
    based on the provided configuration.

    Args:
        config (Config): An instance of a configuration class 
        containing model parameters and settings. 
                         It should include attributes such as `model_name`, 
                         `bnb_4bit_compute_dtype`, `use_4bit`,
                         `bnb_4bit_quant_type`, `use_nested_quant`, and others as required.

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The loaded pre-trained language model.
            - tokenizer (AutoTokenizer): The tokenizer associated with the model.

    Notes:
        - The function checks GPU compatibility with bfloat16 if 
        `use_4bit` is enabled and adjusts configuration settings accordingly.
        - The `BitsAndBytesConfig` is used to configure quantization settings for the model.
        - The function handles different model names with specific loading configurations.
    """
    device_map = "auto"

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16: accelerate training with --bf16"
            )
            print("=" * 80)

            config.fp16 = False
            config.bf16 = True
            config.tf32 = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Load base model
    if config.model_name == "vinai/PhoGPT-7B5-Instruct":
        cfg = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=cfg,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

    elif config.model_name == "vilm/vietcuna-7b-v3":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        model.config.use_cache = True
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        if "gpt" in config.model_name:
            tokenizer.padding_side = "left"

    return model, tokenizer
