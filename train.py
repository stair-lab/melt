# Install the following libraries:
# pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 scipy

from dataclasses import dataclass, field
from typing import Optional

import os
import torch
import neptune
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "The model that you want to train from the Hugging Face hub"},
    )
    dataset_name: Optional[str] = field(
        default="vietgpt/wikipedia_vi",
        metadata={"help": "The instruction dataset to use"},
    )
    new_model: Optional[str] = field(
        default="martinakaduc/llama-2-7b-hf-vi", metadata={"help": "Fine-tuned model name"}
    )
    merge_and_push: Optional[bool] = field(
        default=True, metadata={"help": "Merge and push weights after training"}
    )

    # QLoRA parameters
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "LoRA attention dimension"}
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
        default=True, metadata={"help": "Enable fp16 training"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Enable bf16 training"}
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable the TF32 mode (available in Ampere and newer GPUs)"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=32,
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
        default=64, metadata={"help": "Save checkpoint every X updates steps"}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps"}
    )
    resume_from_checkpoint: bool = field(
        default=False, 
        metadata={"help": "Allows to resume training from the latest checkpoint in output_dir"}
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


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
device_map = {"": 0}

# Load dataset (you can process it here)
if script_args.dataset_name == "oscar-corpus/OSCAR-2301":
    subset_name = "vi"
else:
    subset_name = None
    
dataset = load_dataset(script_args.dataset_name, 
                       name=subset_name, 
                       split="train",
                       num_proc=os.cpu_count()-2)

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.use_4bit,
    bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=script_args.use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and script_args.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with --bf16")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    r=script_args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    save_total_limit=150,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    weight_decay=script_args.weight_decay,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    dataloader_num_workers=os.cpu_count()-2,
    push_to_hub=False,
    report_to="none",
)

neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
run = neptune.init_run(project=os.environ["NEPTUNE_PROJECT"], api_token=neptune_api_token)
neptune_monitor = transformers.integrations.NeptuneCallback(run=run, log_parameters=False)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
    callbacks=[neptune_monitor]
)

# Train model
trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

# Save trained model
trainer.model.save_pretrained(script_args.new_model)

if script_args.merge_and_push:
    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.new_model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()

    model.push_to_hub(script_args.new_model, use_temp_dir=False)
    tokenizer.push_to_hub(script_args.new_model, use_temp_dir=False)