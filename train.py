import os

import neptune
import torch
import transformers
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from script_arguments import ScriptArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaConfig,
    TrainingArguments,
)
from trl import SFTTrainer

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
device_map = "auto"

# Load dataset (you can process it here)
if script_args.dataset_name == "oscar-corpus/OSCAR-2301":
    subset_name = "vi"
else:
    subset_name = None

dataset = load_dataset(
    script_args.dataset_name, name=subset_name, split="train", num_proc=8
)

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
if script_args.scratch:
    configuration = LlamaConfig(
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model = AutoModelForCausalLM.from_config(configuration)
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    script_args.tokenizer_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load LoRA configuration
if script_args.use_lora:
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    save_total_limit=100,
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
    dataloader_num_workers=8,
    push_to_hub=False,
    report_to="none",
    # load_best_model_at_end=True
)

neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
run = neptune.init_run(
    project=os.environ["NEPTUNE_PROJECT"], api_token=neptune_api_token
)
neptune_monitor = transformers.integrations.NeptuneCallback(
    run=run, log_parameters=False
)

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
    callbacks=[neptune_monitor],
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
