import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_model(config):
    device_map = "auto"

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with --bf16")
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
        cfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
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
