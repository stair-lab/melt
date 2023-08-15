import torch
from peft import AutoPeftModelForCausalLM

model_name = "martinakaduc/llama-2-7b-hf-vi"
new_model_name = "martinakaduc/llama-2-7b-hf-vi-merged"
model = AutoPeftModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
model = model.merge_and_unload()
model.save_pretrained(new_model_name)
