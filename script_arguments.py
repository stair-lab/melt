from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "The model that you want to train from the Hugging Face hub"},
    )
    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "The tokenizer from the Hugging Face hub"},
    )
    dataset_name: Optional[str] = field(
        default="vietgpt/wikipedia_vi",
        metadata={"help": "The instruction dataset to use"},
    )
    new_model: Optional[str] = field(
        default="martinakaduc/llama-2-7b-hf-vi",
        metadata={"help": "Fine-tuned model name"},
    )
    scratch: Optional[bool] = field(
        default=False, metadata={"help": "Wheather train from scratch"}
    )
    merge_and_push: Optional[bool] = field(
        default=False, metadata={"help": "Merge and push weights after training"}
    )

    # QLoRA parameters
    use_lora: Optional[bool] = field(
        default=True, metadata={"help": "Enable LoRA fine-tuning"}
    )
    lora_r: Optional[int] = field(
        default=128, metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability for LoRA layers"}
    )

    # bitsandbytes parameters
    use_4bit: Optional[bool] = field(
        default=True, metadata={"help": "Activate 4-bit precision base model loading"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Activate nested quantization for 4-bit base models (double quantization)"
        },
    )

    # TrainingArguments parameters
    output_dir: str = field(
        default="./results",
        metadata={
            "help": "Output directory where the model predictions and checkpoints will be stored"
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Enable fp16 training"}
    )
    bf16: Optional[bool] = field(
        default=True, metadata={"help": "Enable bf16 training"}
    )
    tf32: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable the TF32 mode (available in Ampere and newer GPUs)"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=12, metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Number of update steps to accumulate the gradients for"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    max_grad_norm: Optional[float] = field(
        default=0.3, metadata={"help": "Maximum gradient normal (gradient clipping)"}
    )
    learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "Initial learning rate (AdamW optimizer)"}
    )
    weight_decay: Optional[int] = field(
        default=0.001,
        metadata={
            "help": "Weight decay to apply to all layers except bias/LayerNorm weights"
        },
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "Optimizer to use"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule"},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Number of training steps (overrides num_train_epochs)"},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={
            "help": "Ratio of steps for a linear warmup (from 0 to learning rate)"
        },
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably"
        },
    )
    save_steps: float = field(
        default=65, metadata={"help": "Save checkpoint every X updates steps"}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps"}
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Allows to resume training from the latest checkpoint in output_dir"
        },
    )

    # SFT parameters
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "Maximum sequence length to use"}
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Pack multiple short examples in the same input sequence to increase efficiency"
        },
    )

    # Inference parameters
    prompting_strategy: Optional[int] = field(
        default=0, metadata={"help": "Prompting strategy to use"}
    )
    continue_infer: Optional[bool] = field(
        default=False,
        metadata={"help": "Wheather to continue previous inference process"},
    )
