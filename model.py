import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model(config):
    device_map = "auto"

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with --bf16")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    if 'gpt' in config.model_name:
        tokenizer.padding_side = 'left'

    return model, tokenizer
